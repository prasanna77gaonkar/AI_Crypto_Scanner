"""Technical indicator calculations used by the scanner."""


def calculate_ema(prices: list[float], period: int = 14) -> float | None:
    if len(prices) < period:
        return None

    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return round(ema, 2)


def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    if len(prices) <= period:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, len(prices)):
        change = prices[index] - prices[index - 1]
        gains.append(change if change > 0 else 0)
        losses.append(abs(change) if change < 0 else 0)

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calculate_macd(prices: list[float]) -> float | None:
    if len(prices) < 26:
        return None

    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    return round(ema12 - ema26, 2)
