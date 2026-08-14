"""Signal scoring based on RSI, EMA, and MACD."""


def generate_signal(rsi: float, ema: float, macd: float, current_price: float) -> dict:
    """Preserve the existing technical signal scoring logic."""
    score = 0

    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2

    if current_price > ema:
        score += 1
    elif current_price < ema:
        score -= 1

    if macd > 0:
        score += 1
    elif macd < 0:
        score -= 1

    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    technical_confidence = min(abs(score) * 20 + 20, 100)
    return {"signal": signal, "score": score, "confidence": technical_confidence}
