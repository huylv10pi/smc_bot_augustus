import os
import sys

os.environ["SMC_CREDIT"] = "0"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import aiohttp
import ccxt
import numpy as np
import pandas as pd
from smartmoneyconcepts import smc

# =====================================================================
# 1. CẤU HÌNH BOT ĐA LUỒNG (MULTI-SYMBOL & MULTI-TIMEFRAME)
# =====================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")


# Danh sách các cặp coin theo dõi đồng thời
SYMBOLS = [
    "BTC/USDT",
    "PAXG/USDT,
]

# Có thể chạy 1 hoặc nhiều khung thời gian cùng lúc, ví dụ: ["5m"] hoặc ["5m", "15m"]
TIMEFRAMES   = ["5m"]
MARKET_TYPE  = "future"     # "future" (Binance USD-M Futures) hoặc "spot"

SWING_LENGTH = 6            # Khớp với thông số trong testsystem.py
RISK_REWARD  = 2.0          # Tỷ lệ R:R = 1:2
RISK_PCT     = 0.02         # Rủi ro 2% mỗi lệnh
HISTORY_BARS = 1000         # Số nến lịch sử lưu giữ cho mỗi cặp
REPLAY_LOOKBACK_BARS = 200  # Số nến đóng gần nhất mỗi cặp dùng để quét lại lệnh đang chạy khi mới bật bot
MAX_WORKERS  = 8            # Số luồng CPU tính toán SMC song song khi nhiều nến đóng cùng lúc

STATE_FILE   = Path(__file__).with_name("live_multi_active_trades.json")


def ts_to_ms(ts: pd.Timestamp) -> int:
    """Chuyển đổi pandas Timestamp (UTC naive hoặc UTC aware) sang epoch milliseconds."""
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms"))


def parse_utc_ts(val) -> pd.Timestamp:
    """Chuẩn hóa giá trị thời gian (str hoặc Timestamp) về pandas Timestamp UTC naive."""
    ts = pd.to_datetime(val)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


# =====================================================================
# 2. CẤU TRÚC TRẠNG THÁI CHO TỪNG CẶP (SYMBOL + TIMEFRAME)
# =====================================================================
@dataclass
class MarketTracker:
    symbol: str
    timeframe: str
    stream_name: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    active_trade: dict | None = None
    last_closed_open_ms: int = -1
    current_price: float = 0.0
    tf_ms: int = 300_000
    calculating: bool = False
    resyncing: bool = False
    wins: int = 0
    losses: int = 0
    cum_pnl_pct: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.timeframe}"


# =====================================================================
# 3. LƯU / KHÔI PHỤC TRẠNG THÁI & LỆNH ĐANG MỞ KHI KHỞI ĐỘNG LẠI BOT
# =====================================================================
def save_active_trades(trackers: dict[str, MarketTracker]) -> None:
    """Lưu an toàn trạng thái của toàn bộ các cặp đang theo dõi ra file JSON."""
    trackers_payload = {}
    for tr in trackers.values():
        trade_copy = None
        if tr.active_trade is not None:
            trade_copy = dict(tr.active_trade)
            trade_copy["candle_time"] = str(parse_utc_ts(trade_copy["candle_time"]))
        trackers_payload[tr.key] = {
            "active_trade": trade_copy,
            "last_closed_open_ms": int(tr.last_closed_open_ms),
            "wins": int(tr.wins),
            "losses": int(tr.losses),
            "cum_pnl_pct": float(tr.cum_pnl_pct),
        }

    payload = {
        "version": 2,
        "config": {
            "market_type": MARKET_TYPE,
            "swing_length": SWING_LENGTH,
            "risk_reward": RISK_REWARD,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "trackers": trackers_payload,
    }
    tmp_path = STATE_FILE.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        print(f"[Cảnh báo] Không thể lưu trạng thái lệnh: {e}")


def load_active_trades(trackers: dict[str, MarketTracker]) -> int:
    """Đọc lại trạng thái lệnh đang mở & mốc nến đã xử lý từ file JSON (nếu khớp cấu hình)."""
    if not STATE_FILE.exists():
        return 0
    restored = 0
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        by_key = {tr.key: tr for tr in trackers.values()}

        # Định dạng mới (version 2) có kiểm tra khớp cấu hình chiến lược
        if isinstance(data, dict) and data.get("version") == 2:
            cfg = data.get("config", {})
            if (
                cfg.get("market_type") != MARKET_TYPE
                or cfg.get("swing_length") != SWING_LENGTH
                or cfg.get("risk_reward") != RISK_REWARD
            ):
                print("[Khởi tạo] Thông số cấu hình đã thay đổi so với file lưu cũ — sẽ quét lại từ lịch sử.")
                return 0

            saved_trackers = data.get("trackers", {})
            for key, item in saved_trackers.items():
                if key not in by_key or not isinstance(item, dict):
                    continue
                tr = by_key[key]
                tr.last_closed_open_ms = int(item.get("last_closed_open_ms", -1))
                tr.wins = int(item.get("wins", 0))
                tr.losses = int(item.get("losses", 0))
                tr.cum_pnl_pct = float(item.get("cum_pnl_pct", 0.0))
                trade = item.get("active_trade")
                if isinstance(trade, dict) and "candle_time" in trade:
                    trade["candle_time"] = str(parse_utc_ts(trade["candle_time"]))
                    tr.active_trade = trade
                    restored += 1
        # Tương thích ngược với định dạng cũ (chỉ lưu dict {key: trade})
        elif isinstance(data, dict):
            for key, trade in data.items():
                if key in by_key and isinstance(trade, dict) and "candle_time" in trade:
                    trade["candle_time"] = str(parse_utc_ts(trade["candle_time"]))
                    by_key[key].active_trade = trade
                    restored += 1
    except Exception as e:
        print(f"[Cảnh báo] Không đọc được file trạng thái cũ ({e}), sẽ quét lại từ lịch sử.")
    return restored


# =====================================================================
# 4. HÀNG ĐỢI BẮN TIN TELEGRAM BẤT ĐỒNG BỘ (CHỐNG NGHẼN & RATE-LIMIT)
# =====================================================================
async def telegram_sender_worker(queue: asyncio.Queue):
    """Gửi Telegram có xác nhận API, retry và log rõ trạng thái."""
    configured = bool(
        TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
        and "YOUR_BOT_TOKEN" not in str(TELEGRAM_BOT_TOKEN)
        and "YOUR_CHAT_ID" not in str(TELEGRAM_CHAT_ID)
    )
    if not configured:
        print("[TELEGRAM] Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID. Chỉ chạy PREVIEW.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            message = await queue.get()
            try:
                if not configured:
                    print(f"\n[TELEGRAM PREVIEW - NOT SENT]\n{message}\n")
                    continue

                payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                delivered = False
                for attempt in range(1, 4):
                    try:
                        async with session.post(url, json=payload) as resp:
                            body_text = await resp.text()
                            try:
                                result = json.loads(body_text)
                            except json.JSONDecodeError:
                                result = {}

                            if resp.status == 200 and result.get("ok") is True:
                                delivered = True
                                msg_id = result.get("result", {}).get("message_id", "?")
                                print(f"[TELEGRAM OK] Gửi thành công | message_id={msg_id}")
                                break
                            if resp.status == 429:
                                wait_sec = result.get("parameters", {}).get("retry_after", 3)
                                print(f"[TELEGRAM RATE LIMIT] Chờ {wait_sec}s (lần {attempt}/3)")
                                await asyncio.sleep(float(wait_sec) + 0.5)
                                continue

                            print(f"[TELEGRAM ERROR] HTTP {resp.status} | "
                                  f"API ok={result.get('ok')} | {body_text[:1000]}")
                            if 400 <= resp.status < 500:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                        print(f"[TELEGRAM NETWORK ERROR] Lần {attempt}/3: {exc}")
                    except Exception as exc:
                        print(f"[TELEGRAM UNEXPECTED ERROR] Lần {attempt}/3: {exc}")

                    if attempt < 3:
                        await asyncio.sleep(attempt * 2)

                if not delivered:
                    print("[TELEGRAM FAILED] Không gửi được sau tối đa 3 lần. "
                          "Kiểm tra token, chat_id, quyền bot và HTML parse_mode.")
                await asyncio.sleep(0.35)
            finally:
                queue.task_done()


# =====================================================================
# 5. TẢI DỮ LIỆU LỊCH SỬ ĐA LUỒNG & TÍNH SMC TRÊN THREAD POOL
# =====================================================================
def fetch_candles_with_live_sync(
    symbol: str, timeframe: str, limit: int = HISTORY_BARS
) -> tuple[pd.DataFrame, dict | None, int]:
    """
    Tải các cây nến ĐÃ ĐÓNG gần nhất kèm cây nến ĐANG MỞ hiện tại cho 1 cặp
    (chạy song song trên ThreadPoolExecutor, có tự thử lại khi mạng chập chờn).
    """
    exchange_cls = ccxt.binanceusdm if MARKET_TYPE == "future" else ccxt.binance
    exchange = exchange_cls({"enableRateLimit": True})

    max_api_limit = 1500 if MARKET_TYPE == "future" else 1000
    fetch_limit = min(limit + 1, max_api_limit)

    last_err = None
    raw = None
    for attempt in range(3):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
            if raw and len(raw) >= 50:
                break
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    if not raw or len(raw) < 50:
        raise RuntimeError(f"Không đủ dữ liệu lịch sử cho {symbol} ({timeframe}): {last_err}")

    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if int(raw[-1][0]) + tf_ms > now_ms:
        closed_bars = raw[:-1]
        last_raw = raw[-1]
        live_bar = {
            "open_time_ms": int(last_raw[0]),
            "timestamp": pd.to_datetime(int(last_raw[0]), unit="ms"),
            "open": float(last_raw[1]),
            "high": float(last_raw[2]),
            "low": float(last_raw[3]),
            "close": float(last_raw[4]),
            "volume": float(last_raw[5]),
        }
    else:
        closed_bars = raw
        live_bar = None

    df = pd.DataFrame(closed_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df, live_bar, tf_ms


def fetch_closed_candles_sync(symbol: str, timeframe: str, limit: int = HISTORY_BARS) -> pd.DataFrame:
    """Tải nến đã đóng cho 1 cặp (giữ tương thích hàm cũ)."""
    df, _, _ = fetch_candles_with_live_sync(symbol, timeframe, limit=limit)
    return df


def evaluate_smc_on_closed_bars(df: pd.DataFrame) -> dict | None:
    """
    Tính toán SMC cho bản chụp DataFrame đã đóng (chạy trên luồng CPU riêng biệt).
    Logic khớp 100% với MultiSMCStrategy trong testsystem.py, không bị look-ahead bias.
    """
    df_calc = df.copy()
    df_smc = df_calc.reset_index(drop=True)

    swings = smc.swing_highs_lows(df_smc, swing_length=SWING_LENGTH)
    bos_choch = smc.bos_choch(df_smc, swings, close_break=True)
    obs = smc.ob(df_smc, swings, close_mitigation=False)

    bos_shifted = bos_choch["BOS"].shift(SWING_LENGTH)
    choch_shifted = bos_choch["CHOCH"].shift(SWING_LENGTH)
    structure = bos_shifted.combine_first(choch_shifted)

    df_calc["trend"] = structure.ffill().fillna(0).values
    df_calc["ob_dir"] = obs["OB"].shift(SWING_LENGTH).ffill().fillna(0).values
    df_calc["ob_top"] = obs["Top"].shift(SWING_LENGTH).ffill().values
    df_calc["ob_bottom"] = obs["Bottom"].shift(SWING_LENGTH).ffill().values

    last = df_calc.iloc[-1]
    candle_time = df_calc.index[-1]
    price = float(last["close"])
    low = float(last["low"])
    high = float(last["high"])
    trend = float(last["trend"])
    ob_dir = float(last["ob_dir"])
    ob_top = float(last["ob_top"])
    ob_bottom = float(last["ob_bottom"])

    # 1. SETUP MUA (BUY / LONG)
    if trend == 1 and ob_dir == 1 and not np.isnan(ob_top) and not np.isnan(ob_bottom):
        if (low <= ob_top) and (price >= ob_bottom):
            sl_buy = ob_bottom * 0.998
            if sl_buy < price:
                tp_buy = price + (price - sl_buy) * RISK_REWARD
                risk_dist = (price - sl_buy) / price
                fraction_size = min(round(RISK_PCT / risk_dist, 4), 0.95)
                if fraction_size > 0:
                    return {
                        "side": "BUY",
                        "entry": price,
                        "sl": sl_buy,
                        "tp": tp_buy,
                        "ob_top": ob_top,
                        "ob_bottom": ob_bottom,
                        "size_pct": round(fraction_size * 100, 2),
                        "candle_time": str(candle_time),
                    }

    # 2. SETUP BÁN (SELL / SHORT)
    if trend == -1 and ob_dir == -1 and not np.isnan(ob_top) and not np.isnan(ob_bottom):
        if (high >= ob_bottom) and (price <= ob_top):
            sl_sell = ob_top * 1.002
            if sl_sell > price:
                tp_sell = price - (sl_sell - price) * RISK_REWARD
                risk_dist = (sl_sell - price) / price
                fraction_size = min(round(RISK_PCT / risk_dist, 4), 0.95)
                if fraction_size > 0:
                    return {
                        "side": "SELL",
                        "entry": price,
                        "sl": sl_sell,
                        "tp": tp_sell,
                        "ob_top": ob_top,
                        "ob_bottom": ob_bottom,
                        "size_pct": round(fraction_size * 100, 2),
                        "candle_time": str(candle_time),
                    }

    return None


def fmt_price(p: float) -> str:
    """Định dạng số lẻ tự động phù hợp cho cả BTC (giá lớn) và XRP/DOGE (giá nhỏ)."""
    if abs(p) >= 100:
        return f"{p:.2f}"
    if abs(p) >= 1:
        return f"{p:.4f}"
    return f"{p:.6f}"


def check_trade_exit(trade: dict, high: float, low: float) -> tuple[bool, bool, float, float]:
    """
    Kiểm tra xem cây nến/tick (high, low) có chạm SL hoặc TP của lệnh đang mở hay không.
    Trả về: (hit_sl, hit_tp, exit_price, pnl_pct)
    """
    side = trade["side"]
    entry = float(trade["entry"])
    sl = float(trade["sl"])
    tp = float(trade["tp"])

    hit_sl = (side == "BUY" and low <= sl) or (side == "SELL" and high >= sl)
    hit_tp = (side == "BUY" and high >= tp) or (side == "SELL" and low <= tp)

    if not (hit_sl or hit_tp):
        return False, False, 0.0, 0.0

    # Nếu 1 nến quét cả SL và TP trong lúc mất mạng, ưu tiên tính SL để bảo toàn quản trị rủi ro
    if hit_sl:
        exit_price = sl
        hit_tp = False
    else:
        exit_price = tp

    pnl_pct = (
        ((exit_price - entry) / entry * 100.0)
        if side == "BUY"
        else ((entry - exit_price) / entry * 100.0)
    )
    return hit_sl, hit_tp, exit_price, pnl_pct


def calc_unrealized_pnl_pct(trade: dict, current_price: float) -> float:
    """Tính % PnL tạm tính của lệnh đang mở theo giá hiện tại."""
    entry = float(trade["entry"])
    if trade["side"] == "BUY":
        return (current_price - entry) / entry * 100.0
    return (entry - current_price) / entry * 100.0


def record_and_format_exit_alert(
    tr: MarketTracker,
    trade: dict,
    exit_price: float,
    pnl_pct: float,
    hit_tp: bool,
    exit_time_str: str,
    note: str = "",
) -> str:
    """Cập nhật thống kê thắng/thua cho cặp giao dịch và định dạng tin nhắn đóng lệnh."""
    if hit_tp:
        tr.wins += 1
        status_icon = "✅ CHỐT LỜI (TP)"
    else:
        tr.losses += 1
        status_icon = "❌ DỪNG LỖ (SL)"
    tr.cum_pnl_pct += pnl_pct

    total_closed = tr.wins + tr.losses
    winrate = (tr.wins / total_closed * 100.0) if total_closed > 0 else 0.0
    note_suffix = f" <i>{note}</i>" if note else ""

    return (
        f"{status_icon} — <b>{tr.symbol} ({tr.timeframe})</b>{note_suffix}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• Lệnh: <b>{trade['side']}</b> | Entry: <code>{fmt_price(float(trade['entry']))}</code> → Exit: <code>{fmt_price(exit_price)}</code>\n"
        f"• Kết quả lệnh: <b>{pnl_pct:+.2f}%</b>\n"
        f"• Thống kê cặp này: <code>{tr.wins}W - {tr.losses}L ({winrate:.0f}%) | {tr.cum_pnl_pct:+.2f}%</code>\n"
        f"• Thời gian: <code>{exit_time_str}</code>"
    )


def format_entry_alert(tr: MarketTracker, signal: dict, bar_ts: pd.Timestamp) -> str:
    """Định dạng tin nhắn khi có tín hiệu vào lệnh mới."""
    icon = "🟢 TÍN HIỆU MUA (BUY)" if signal["side"] == "BUY" else "🔴 TÍN HIỆU BÁN (SELL)"
    return (
        f"{icon} — <b>{tr.symbol} ({tr.timeframe})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"• <b>Entry:</b> <code>{fmt_price(float(signal['entry']))}</code>\n"
        f"• <b>Stop Loss (SL):</b> <code>{fmt_price(float(signal['sl']))}</code>\n"
        f"• <b>Take Profit (TP):</b> <code>{fmt_price(float(signal['tp']))}</code>\n"
        f"• <b>Vùng OB:</b> <code>{fmt_price(float(signal['ob_bottom']))} - {fmt_price(float(signal['ob_top']))}</code>\n"
        f"• <b>Khối lượng gợi ý:</b> <code>{signal['size_pct']}% NAV</code> (R:R 1:{RISK_REWARD})\n"
        f"• <b>Nến xác nhận:</b> <code>{bar_ts} UTC</code>"
    )


def format_single_active_trade_block(
    idx: int, tr: MarketTracker, is_missed_entry: bool = False
) -> str:
    """Định dạng khối thông tin cho 1 cặp đang có lệnh chạy."""
    trade = tr.active_trade
    side_label = "🟢 BUY (LONG)" if trade["side"] == "BUY" else "🔴 SELL (SHORT)"
    cur_p = tr.current_price if tr.current_price > 0 else float(trade["entry"])
    unrealized_pct = calc_unrealized_pnl_pct(trade, cur_p)
    missed_tag = " <i>(Mới khớp khi mất kết nối)</i>" if is_missed_entry else ""

    return (
        f"<b>{idx}. {tr.symbol} ({tr.timeframe})</b> — <b>{side_label}</b>{missed_tag}\n"
        f"• <b>Entry:</b> <code>{fmt_price(float(trade['entry']))}</code> | "
        f"<b>Hiện tại:</b> <code>{fmt_price(cur_p)}</code> (<b>{unrealized_pct:+.2f}%</b>)\n"
        f"• <b>SL:</b> <code>{fmt_price(float(trade['sl']))}</code> | "
        f"<b>TP:</b> <code>{fmt_price(float(trade['tp']))}</code>\n"
        f"• <b>Vùng OB:</b> <code>{fmt_price(float(trade['ob_bottom']))} - {fmt_price(float(trade['ob_top']))}</code>\n"
        f"• <b>Khối lượng gợi ý:</b> <code>{trade['size_pct']}% NAV</code> (R:R 1:{RISK_REWARD})\n"
        f"• <b>Nến xác nhận:</b> <code>{trade['candle_time']} UTC</code>"
    )


def format_active_trades_messages(
    active_items: list[tuple[MarketTracker, bool]],
    header: str,
) -> list[str]:
    """
    Tạo danh sách tin nhắn Telegram liệt kê toàn bộ các lệnh đang diễn ra
    (tự chia trang nếu vượt giới hạn độ dài tin nhắn của Telegram).
    """
    if not active_items:
        return []

    messages: list[str] = []
    current_lines = [header, "━━━━━━━━━━━━━━━━━━"]
    current_len = sum(len(x) + 1 for x in current_lines)

    for idx, (tr, is_missed) in enumerate(active_items, start=1):
        block = format_single_active_trade_block(idx, tr, is_missed_entry=is_missed)
        if current_len + len(block) + 2 > 3500 and len(current_lines) > 2:
            messages.append("\n".join(current_lines))
            current_lines = [f"{header} <i>(tiếp)</i>", "━━━━━━━━━━━━━━━━━━", block]
            current_len = sum(len(x) + 1 for x in current_lines)
        else:
            if len(current_lines) > 2:
                current_lines.append("")
                current_len += 1
            current_lines.append(block)
            current_len += len(block) + 1

    if len(current_lines) > 2:
        messages.append("\n".join(current_lines))
    return messages


def sync_one_tracker_sync(
    symbol: str,
    timeframe: str,
    active_trade: dict | None,
    last_closed_open_ms: int,
) -> dict:
    """
    Hàm đồng bộ & quét khôi phục lệnh chạy độc lập trên worker thread cho 1 cặp (Symbol + Timeframe):
    - Khi mới khởi chạy (chưa có lịch sử xử lý): quét lại REPLAY_LOOKBACK_BARS nến đóng gần nhất
      + nến đang mở hiện tại để phát hiện bất kỳ Entry nào đã khớp và hiện vẫn đang chạy.
    - Khi kết nối lại sau gián đoạn: quét bù toàn bộ các nến đóng bị bỏ lỡ + nến đang mở hiện tại
      để phát hiện lệnh chạm SL/TP hoặc Entry mới xuất hiện trong lúc mất kết nối.
    """
    df, live_bar, tf_ms = fetch_candles_with_live_sync(symbol, timeframe, limit=HISTORY_BARS)
    closed_outcomes: list[dict] = []

    trade_origin = "existing" if active_trade is not None else "none"

    if last_closed_open_ms > 0:
        last_processed_ts = pd.to_datetime(last_closed_open_ms, unit="ms")
    elif active_trade is not None and "candle_time" in active_trade:
        # Hỗ trợ trường hợp đọc từ file trạng thái định dạng cũ chưa có last_closed_open_ms
        last_processed_ts = parse_utc_ts(active_trade["candle_time"])
    else:
        last_processed_ts = None

    # Trường hợp 1: Khởi động mới hoàn toàn hoặc mất kết nối quá lâu ngoài phạm vi lịch sử
    if last_processed_ts is None or last_processed_ts < df.index[0]:
        active_trade = None
        trade_origin = "none"

        start_idx = max(SWING_LENGTH * 4, len(df) - REPLAY_LOOKBACK_BARS)
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            h, l = float(row["high"]), float(row["low"])

            if active_trade is not None:
                hit_sl, hit_tp, _, _ = check_trade_exit(active_trade, h, l)
                if hit_sl or hit_tp:
                    # Đây là lệnh giả lập tái dựng từ lịch sử, không phải trạng thái
                    # lệnh đã được bot xác nhận trước đó; không phát cảnh báo đóng lệnh cũ.
                    active_trade = None
                    trade_origin = "none"

            if active_trade is None:
                sig = evaluate_smc_on_closed_bars(df.iloc[: idx + 1])
                if sig is not None:
                    active_trade = sig
                    trade_origin = "reconstructed"

        # Nếu lệnh vừa khôi phục bị quét SL/TP bởi cây nến đang mở ngay trước khi lên danh sách khởi động
        if (
            active_trade is not None
            and live_bar is not None
            and live_bar["timestamp"] > parse_utc_ts(active_trade["candle_time"])
        ):
            hit_sl, hit_tp, _, _ = check_trade_exit(
                active_trade, float(live_bar["high"]), float(live_bar["low"])
            )
            if hit_sl or hit_tp:
                # Chỉ loại lệnh tái dựng; không phát cảnh báo cho giao dịch lịch sử.
                active_trade = None
                trade_origin = "none"

    # Trường hợp 2: Kết nối lại hoặc tiếp nối từ STATE_FILE -> quét bù các nến bị bỏ lỡ
    else:
        missed_positions = np.flatnonzero(df.index > last_processed_ts)
        for idx in missed_positions:
            bar_ts = df.index[idx]
            row = df.iloc[idx]
            h, l = float(row["high"]), float(row["low"])

            if active_trade is not None and bar_ts > parse_utc_ts(active_trade["candle_time"]):
                hit_sl, hit_tp, exit_price, pnl_pct = check_trade_exit(active_trade, h, l)
                if hit_sl or hit_tp:
                    # Luôn gửi thông báo SL/TP đối với mọi lệnh trong giai đoạn bù gap
                    closed_outcomes.append(
                        {
                            "trade": active_trade,
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "hit_tp": hit_tp,
                            "exit_time_str": f"{bar_ts} UTC",
                            "note": "(Khớp trong lúc mất kết nối)",
                        }
                    )
                    active_trade = None
                    trade_origin = "none"

            if active_trade is None:
                sig = evaluate_smc_on_closed_bars(df.iloc[: idx + 1])
                if sig is not None:
                    active_trade = sig
                    trade_origin = "missed_candle"

        # Kiểm tra tiếp với râu High/Low của cây nến hiện tại đang mở (chưa đóng)
        if (
            active_trade is not None
            and live_bar is not None
            and live_bar["timestamp"] > parse_utc_ts(active_trade["candle_time"])
        ):
            hit_sl, hit_tp, exit_price, pnl_pct = check_trade_exit(
                active_trade, float(live_bar["high"]), float(live_bar["low"])
            )
            if hit_sl or hit_tp:
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                closed_outcomes.append(
                    {
                        "trade": active_trade,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "hit_tp": hit_tp,
                        "exit_time_str": now_str,
                        "note": "(Khớp khi đồng bộ lại)",
                    }
                )
                active_trade = None
                trade_origin = "none"

    current_price = float(live_bar["close"]) if live_bar is not None else float(df["close"].iloc[-1])

    return {
        "df": df,
        "active_trade": active_trade,
        "last_closed_open_ms": ts_to_ms(df.index[-1]),
        "current_price": current_price,
        "tf_ms": tf_ms,
        "closed_outcomes": closed_outcomes,
        "trade_origin": trade_origin,
    }


# =====================================================================
# 6. XỬ LÝ TÍN HIỆU ĐÓNG NẾN & BÙ GAP BẤT ĐỒNG BỘ TRÊN LUỒNG RIÊNG
# =====================================================================
async def process_closed_bar_signal(
    tracker: MarketTracker,
    df_snapshot: pd.DataFrame,
    bar_ts: pd.Timestamp,
    executor: ThreadPoolExecutor,
    tg_queue: asyncio.Queue,
    all_trackers: dict[str, MarketTracker],
):
    tracker.calculating = True
    try:
        loop = asyncio.get_running_loop()
        signal = await loop.run_in_executor(executor, evaluate_smc_on_closed_bars, df_snapshot)

        if signal is not None and tracker.active_trade is None:
            tracker.active_trade = signal
            save_active_trades(all_trackers)

            alert_entry = format_entry_alert(tracker, signal, bar_ts)
            print(f"\n{alert_entry}")
            await tg_queue.put(alert_entry)
    except Exception as e:
        print(f"[Lỗi tính SMC {tracker.symbol} {tracker.timeframe}]: {e}")
    finally:
        tracker.calculating = False


async def resync_single_tracker_gap(
    tracker: MarketTracker,
    executor: ThreadPoolExecutor,
    tg_queue: asyncio.Queue,
    all_trackers: dict[str, MarketTracker],
):
    """Đồng bộ bù riêng cho 1 stream nếu phát hiện nhảy cóc nến ngay lúc bắt tay WebSocket."""
    if tracker.resyncing:
        return
    tracker.resyncing = True
    try:
        loop = asyncio.get_running_loop()
        trade_snapshot = dict(tracker.active_trade) if tracker.active_trade else None
        res = await loop.run_in_executor(
            executor,
            sync_one_tracker_sync,
            tracker.symbol,
            tracker.timeframe,
            trade_snapshot,
            tracker.last_closed_open_ms,
        )
        tracker.df = res["df"]
        tracker.active_trade = res["active_trade"]
        tracker.last_closed_open_ms = res["last_closed_open_ms"]
        tracker.current_price = res["current_price"]
        tracker.tf_ms = res["tf_ms"]

        for outcome in res["closed_outcomes"]:
            alert_exit = record_and_format_exit_alert(
                tracker,
                outcome["trade"],
                outcome["exit_price"],
                outcome["pnl_pct"],
                outcome["hit_tp"],
                outcome["exit_time_str"],
                note=outcome["note"],
            )
            print(f"\n{alert_exit}")
            await tg_queue.put(alert_exit)

        if tracker.active_trade is not None and res["trade_origin"] == "missed_candle":
            msgs = format_active_trades_messages(
                [(tracker, True)],
                f"🚨 <b>PHÁT HIỆN ENTRY ĐANG CHẠY — {tracker.symbol} ({tracker.timeframe})</b>",
            )
            for m in msgs:
                print(f"\n{m}")
                await tg_queue.put(m)

        save_active_trades(all_trackers)
    except Exception as e:
        print(f"[Lỗi đồng bộ bù {tracker.symbol} {tracker.timeframe}]: {e}")
    finally:
        tracker.resyncing = False


# =====================================================================
# 7. ĐỒNG BỘ LỊCH SỬ SONG SONG & LẮNG NGHE COMBINED WEBSOCKET STREAM
# =====================================================================
async def sync_all_histories(
    trackers: dict[str, MarketTracker],
    executor: ThreadPoolExecutor,
) -> dict:
    """
    Tải song song dữ liệu mới nhất & khôi phục trạng thái lệnh đang chạy cho toàn bộ các cặp.
    Trả về dict gồm exit_alerts (các lệnh đã đóng trong lúc mất mạng) và missed_keys (các cặp mới vào Entry).
    """
    loop = asyncio.get_running_loop()

    async def _sync_one(tr: MarketTracker) -> tuple[MarketTracker, dict]:
        trade_snapshot = dict(tr.active_trade) if tr.active_trade else None
        res = await loop.run_in_executor(
            executor,
            sync_one_tracker_sync,
            tr.symbol,
            tr.timeframe,
            trade_snapshot,
            tr.last_closed_open_ms,
        )
        return tr, res

    results = await asyncio.gather(*(_sync_one(tr) for tr in trackers.values()))

    exit_alerts: list[str] = []
    missed_keys: set[str] = set()

    for tr, res in results:
        tr.df = res["df"]
        tr.active_trade = res["active_trade"]
        tr.last_closed_open_ms = res["last_closed_open_ms"]
        tr.current_price = res["current_price"]
        tr.tf_ms = res["tf_ms"]

        for outcome in res["closed_outcomes"]:
            exit_alerts.append(
                record_and_format_exit_alert(
                    tr,
                    outcome["trade"],
                    outcome["exit_price"],
                    outcome["pnl_pct"],
                    outcome["hit_tp"],
                    outcome["exit_time_str"],
                    note=outcome["note"],
                )
            )

        if tr.active_trade is not None and res["trade_origin"] == "missed_candle":
            missed_keys.add(tr.key)

        pos_label = tr.active_trade["side"] if tr.active_trade else "Trống"
        print(
            f"  ✔ {tr.symbol:<10} [{tr.timeframe:<3}] — {len(tr.df)} nến | "
            f"Giá: {fmt_price(tr.current_price):<10} | Vị thế: {pos_label}"
        )

    save_active_trades(trackers)
    return {
        "exit_alerts": exit_alerts,
        "missed_keys": missed_keys,
    }


async def run_multi_bot():
    # Khởi tạo bộ theo dõi cho mọi tổ hợp (Symbol x Timeframe)
    trackers: dict[str, MarketTracker] = {}
    for sym in SYMBOLS:
        clean_sym = sym.replace("/", "").lower()
        for tf in TIMEFRAMES:
            stream_name = f"{clean_sym}@kline_{tf}"
            trackers[stream_name] = MarketTracker(symbol=sym, timeframe=tf, stream_name=stream_name)

    restored_count = load_active_trades(trackers)
    if restored_count > 0:
        print(f"[Khôi phục] Đọc thấy {restored_count} lệnh đang mở từ file trạng thái trước đó.")

    tg_queue: asyncio.Queue = asyncio.Queue()
    print(
        "[CONFIG] Telegram token: "
        + ("đã cấu hình" if TELEGRAM_BOT_TOKEN and "YOUR_BOT_TOKEN" not in TELEGRAM_BOT_TOKEN else "THIẾU")
        + " | Chat ID: "
        + ("đã cấu hình" if TELEGRAM_CHAT_ID and "YOUR_CHAT_ID" not in TELEGRAM_CHAT_ID else "THIẾU")
    )
    tg_task = asyncio.create_task(telegram_sender_worker(tg_queue))

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="smc_worker")
    bg_tasks: set[asyncio.Task] = set()

    def _spawn_bg_task(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        return task

    streams_query = "/".join(trackers.keys())
    ws_base = "wss://fstream.binance.com/stream" if MARKET_TYPE == "future" else "wss://stream.binance.com:9443/stream"
    ws_url = f"{ws_base}?streams={streams_query}"

    has_started = False
    last_empty_reconnect_notify_ts = 0.0
    pending_exit_alerts: list[str] = []
    pending_missed_keys: set[str] = set()

    try:
        while True:
            try:
                # Đợi các tác vụ tính SMC nền còn dở hoàn tất trước khi đồng bộ lại
                if bg_tasks:
                    await asyncio.gather(*list(bg_tasks), return_exceptions=True)

                action_label = "Đang tải & quét lịch sử ban đầu" if not has_started else "Đang đồng bộ lại nến & kiểm tra lệnh sau gián đoạn"
                print(
                    f"\n[{action_label}] {len(trackers)} luồng ({len(SYMBOLS)} cặp x {len(TIMEFRAMES)} khung) "
                    f"trên {MAX_WORKERS} luồng CPU..."
                )
                sync_res = await sync_all_histories(trackers, executor)
                pending_exit_alerts.extend(sync_res["exit_alerts"])
                pending_missed_keys.update(sync_res["missed_keys"])
                # Loại bỏ khỏi danh sách missed_keys những cặp hiện không còn giữ lệnh
                pending_missed_keys = {
                    tr.key for tr in trackers.values() if tr.active_trade is not None and tr.key in pending_missed_keys
                }

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url, heartbeat=20) as ws:
                        print(f"[WebSocket Đa Luồng] Đã kết nối {len(trackers)} streams trên 1 kênh gộp. Đang theo dõi realtime...")

                        # 1. Gửi thông báo đóng lệnh bù nếu có lệnh chạm SL/TP trong lúc mất kết nối
                        for alert_msg in pending_exit_alerts:
                            print(f"\n{alert_msg}")
                            await tg_queue.put(alert_msg)
                        pending_exit_alerts.clear()

                        active_items = [
                            (tr, tr.key in pending_missed_keys)
                            for tr in trackers.values()
                            if tr.active_trade is not None
                        ]

                        # 2. Gửi thông báo Khởi động hoặc Kết nối lại kèm chi tiết các lệnh đang chạy
                        if not has_started:
                            has_started = True
                            pending_missed_keys.clear()

                            startup_msg = (
                                f"🚀 <b>SMC Multi-Thread Bot đã khởi động!</b>\n"
                                f"• Thị trường: <code>{MARKET_TYPE.upper()}</code> | Swing: <code>{SWING_LENGTH}</code> | R:R: <code>1:{RISK_REWARD}</code>\n"
                                f"• Các cặp ({len(SYMBOLS)}): <code>{', '.join(SYMBOLS)}</code>\n"
                                f"• Khung giờ: <code>{', '.join(TIMEFRAMES)}</code> | Luồng CPU: <code>{MAX_WORKERS}</code>\n"
                                f"• Đang có lệnh diễn ra: <code>{len(active_items)} lệnh</code>"
                            )
                            print(f"\n{startup_msg}")
                            await tg_queue.put(startup_msg)

                            if active_items:
                                active_msgs = format_active_trades_messages(
                                    active_items,
                                    f"📌 <b>DANH SÁCH LỆNH ĐANG DIỄN RA ({len(active_items)})</b>",
                                )
                                for m in active_msgs:
                                    print(f"\n{m}")
                                    await tg_queue.put(m)
                        else:
                            if active_items:
                                has_new_missed = any(is_missed for _, is_missed in active_items)
                                reconnect_header = (
                                    f"🚨 <b>ĐÃ KẾT NỐI LẠI — PHÁT HIỆN ENTRY MỚI & LỆNH ĐANG CHẠY ({len(active_items)})</b>"
                                    if has_new_missed
                                    else f"🔄 <b>ĐÃ KẾT NỐI LẠI — ĐANG CÓ {len(active_items)} LỆNH DIỄN RA</b>"
                                )
                                pending_missed_keys.clear()
                                active_msgs = format_active_trades_messages(active_items, reconnect_header)
                                for m in active_msgs:
                                    print(f"\n{m}")
                                    await tg_queue.put(m)
                            else:
                                now_epoch = datetime.now(timezone.utc).timestamp()
                                if now_epoch - last_empty_reconnect_notify_ts >= 60:
                                    last_empty_reconnect_notify_ts = now_epoch
                                    reconnect_msg = (
                                        f"🔄 <b>Đã kết nối lại WebSocket Đa Luồng</b> ({len(trackers)} streams)\n"
                                        f"• Trạng thái: <i>Hiện chưa có lệnh đang mở.</i>"
                                    )
                                    print(f"\n{reconnect_msg}")
                                    await tg_queue.put(reconnect_msg)

                        # 3. Lắng nghe dữ liệu nến realtime của toàn bộ các cặp
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue

                            payload = json.loads(msg.data)
                            stream_name = payload.get("stream")
                            event_data = payload.get("data", {})
                            k = event_data.get("k")
                            if not stream_name or not k or stream_name not in trackers:
                                continue

                            tr = trackers[stream_name]
                            live_high = float(k["h"])
                            live_low = float(k["l"])
                            live_close = float(k["c"])
                            is_closed = bool(k["x"])
                            open_time_ms = int(k["t"])
                            tr.current_price = live_close

                            # ---------------------------------------------------------
                            # A. KIỂM TRA CHẠM SL / TP TỨC THỜI CHO TỪNG CẶP ĐANG MỞ LỆNH
                            # (Ưu tiên kiểm tra trước khi xử lý resync gap để không nuốt SL/TP)
                            # ---------------------------------------------------------
                            if tr.active_trade is not None:
                                trade_open_ms = ts_to_ms(parse_utc_ts(tr.active_trade["candle_time"]))
                                if open_time_ms > trade_open_ms:
                                    hit_sl, hit_tp, exit_price, pnl_pct = check_trade_exit(
                                        tr.active_trade, live_high, live_low
                                    )
                                    if hit_sl or hit_tp:
                                        print(
                                            f"[TP/SL HIT] {tr.symbol} {tr.timeframe} "
                                            f"side={tr.active_trade['side']} "
                                            f"high={fmt_price(live_high)} low={fmt_price(live_low)} "
                                            f"SL={fmt_price(float(tr.active_trade['sl']))} "
                                            f"TP={fmt_price(float(tr.active_trade['tp']))}"
                                        )
                                        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                                        alert_exit = record_and_format_exit_alert(
                                            tr,
                                            tr.active_trade,
                                            exit_price,
                                            pnl_pct,
                                            hit_tp,
                                            now_str,
                                        )
                                        print(f"\n{alert_exit}")
                                        tr.active_trade = None
                                        save_active_trades(trackers)
                                        await tg_queue.put(alert_exit)

                            # Phòng trường hợp có nến đóng đúng vào tích tắc bắt tay WebSocket
                            if open_time_ms > tr.last_closed_open_ms + tr.tf_ms and not tr.resyncing:
                                _spawn_bg_task(
                                    resync_single_tracker_gap(
                                        tracker=tr,
                                        executor=executor,
                                        tg_queue=tg_queue,
                                        all_trackers=trackers,
                                    )
                                )
                                continue

                            # ---------------------------------------------------------
                            # B. KHI NẾN ĐÓNG -> ĐẨY SANG THREAD POOL TÍNH SMC SONG SONG
                            # ---------------------------------------------------------
                            if is_closed and open_time_ms > tr.last_closed_open_ms:
                                tr.last_closed_open_ms = open_time_ms
                                bar_ts = pd.to_datetime(open_time_ms, unit="ms")

                                tr.df.loc[bar_ts] = [
                                    float(k["o"]),
                                    live_high,
                                    live_low,
                                    live_close,
                                    float(k["v"]),
                                ]
                                if len(tr.df) > HISTORY_BARS:
                                    tr.df = tr.df.iloc[-HISTORY_BARS:]

                                save_active_trades(trackers)

                                status_str = tr.active_trade["side"] if tr.active_trade else "Trống"
                                print(f"[{bar_ts}] {tr.symbol:<9} ({tr.timeframe}) đóng nến @ {fmt_price(live_close)} | Vị thế: {status_str}")

                                if tr.active_trade is None and not tr.calculating and not tr.resyncing:
                                    _spawn_bg_task(
                                        process_closed_bar_signal(
                                            tracker=tr,
                                            df_snapshot=tr.df.copy(),
                                            bar_ts=bar_ts,
                                            executor=executor,
                                            tg_queue=tg_queue,
                                            all_trackers=trackers,
                                        )
                                    )

                print("[WebSocket Đa Luồng] Kết nối đã đóng. Tự động kết nối & đồng bộ lại sau 3 giây...")
                await asyncio.sleep(3)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Mất kết nối WebSocket] {e}. Tự động kết nối & đồng bộ lại sau 5 giây...")
                await asyncio.sleep(5)
    finally:
        for task in list(bg_tasks):
            task.cancel()
        tg_task.cancel()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    try:
        asyncio.run(run_multi_bot())
    except KeyboardInterrupt:
        print("\nĐã dừng SMC Multi-Thread Bot.")
