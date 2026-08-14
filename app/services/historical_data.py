"""Historical cryptocurrency market data from CoinGecko."""

import httpx


COINGECKO_URL = "https://api.coingecko.com/api/v3/coins"


async def get_historical_data(coin_id: str, days: int = 30) -> list[list[float]]:
    """Return CoinGecko price points as ``[timestamp_ms, price_usd]`` pairs."""
    url = f"{COINGECKO_URL}/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("prices", [])
