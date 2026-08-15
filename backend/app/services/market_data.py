"""Fetch live cryptocurrency prices from CoinGecko's public API."""

from datetime import datetime, timezone
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import HTTPException


COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
}


def get_market_tickers() -> list[dict]:
    """Return the latest USD market data for the supported cryptocurrencies."""
    query = urlencode(
        {
            "ids": ",".join(COINS.values()),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
            "include_last_updated_at": "true",
        }
    )

    try:
        with urlopen(f"{COINGECKO_URL}?{query}", timeout=10) as response:
            market_data = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=502,
            detail="Unable to fetch live market data from CoinGecko.",
        ) from error

    return [
        {
            "symbol": symbol,
            "price": market_data[coin_id]["usd"],
            "change_24h": market_data[coin_id]["usd_24h_change"],
            "volume_24h": market_data[coin_id]["usd_24h_vol"],
            "updated_at": datetime.fromtimestamp(
                market_data[coin_id]["last_updated_at"], tz=timezone.utc
            ).isoformat(),
        }
        for symbol, coin_id in COINS.items()
    ]
