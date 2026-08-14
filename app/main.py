from io import BytesIO
import os

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.services.chart_analyzer import analyze_chart_image
from app.services.historical_data import get_historical_data
from app.services.indicators import calculate_ema, calculate_macd, calculate_rsi
from app.services.market_data import get_market_tickers
from app.services.scanner import generate_signal


app = FastAPI()

_local_frontend_origins = ["http://127.0.0.1:5500", "http://localhost:5500"]
_configured_frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
_allowed_frontend_origins = list(dict.fromkeys(_local_frontend_origins + _configured_frontend_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "online", "message": "AI Crypto Scanner is running"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/market/tickers")
def market_tickers() -> list[dict]:
    return get_market_tickers()


def _prices_from_history(history: list[list[float]]) -> list[float]:
    return [float(item[1]) for item in history if len(item) >= 2]


def _indicator_values(prices: list[float]) -> tuple[float, float, float]:
    if len(prices) < 26:
        raise HTTPException(
            status_code=422,
            detail="At least 26 historical price points are required for RSI, EMA_14, and MACD.",
        )

    rsi = calculate_rsi(prices)
    ema = calculate_ema(prices)
    macd = calculate_macd(prices)
    if rsi is None or ema is None or macd is None:
        raise HTTPException(status_code=422, detail="Unable to calculate technical indicators.")
    return rsi, ema, macd


async def _history_or_upstream_error(coin_id: str, days: int) -> list[list[float]]:
    if not 1 <= days <= 365:
        raise HTTPException(status_code=422, detail="days must be between 1 and 365.")
    try:
        return await get_historical_data(coin_id, days)
    except httpx.HTTPStatusError as error:
        status_code = 404 if error.response.status_code == 404 else 502
        detail = "Coin not found." if status_code == 404 else "Unable to fetch historical market data."
        raise HTTPException(status_code=status_code, detail=detail) from error
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="Unable to fetch historical market data.") from error


@app.get("/api/market/history/{coin_id}")
async def market_history(coin_id: str, days: int = 30) -> dict:
    prices = await _history_or_upstream_error(coin_id, days)
    return {"coin": coin_id, "days": days, "prices": prices}


@app.get("/api/market/indicators/{coin_id}")
async def market_indicators(coin_id: str, days: int = 30) -> dict:
    prices = _prices_from_history(await _history_or_upstream_error(coin_id, days))
    rsi, ema, macd = _indicator_values(prices)
    return {"coin": coin_id, "indicators": {"RSI": rsi, "EMA_14": ema, "MACD": macd}}


async def _scan_coin(coin_id: str, days: int = 30) -> dict:
    prices = _prices_from_history(await _history_or_upstream_error(coin_id, days))
    rsi, ema, macd = _indicator_values(prices)
    current_price = prices[-1]
    scanner = generate_signal(rsi, ema, macd, current_price)
    return {
        "coin": coin_id,
        "current_price": current_price,
        "RSI": rsi,
        "EMA_14": ema,
        "MACD": macd,
        **scanner,
    }


@app.get("/api/scanner/all")
async def scan_all_coins() -> dict:
    coin_ids = ("bitcoin", "ethereum", "solana", "binancecoin", "ripple")
    results = []
    for coin_id in coin_ids:
        try:
            results.append(await _scan_coin(coin_id))
        except HTTPException as error:
            results.append({"coin": coin_id, "status": "error", "message": error.detail})
    return {"total_coins": len(results), "results": results}


@app.get("/api/scanner/{coin_id}")
async def scan_coin(coin_id: str, days: int = 30) -> dict:
    return await _scan_coin(coin_id, days)


def _chart_error(message: str, status_code: int) -> JSONResponse:
    """Return a predictable JSON object even when upload or analysis fails."""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "scanner": {
                "signal": "INVALID",
                "confidence": 0,
                "trend": "UNKNOWN",
                "timeframe": "10m",
                "analysis": message,
            },
        },
    )


@app.post("/api/chart/analyze", response_model=None)
async def analyze_chart(file: UploadFile | None = File(default=None)) -> dict | JSONResponse:
    """Analyze one uploaded chart image. Only validated candlestick charts get a signal."""
    if file is None:
        return _chart_error("No image file was uploaded in the 'file' form field.", 422)
    if not file.content_type or not file.content_type.startswith("image/"):
        return _chart_error("Upload a valid image file.", 415)

    try:
        content = await file.read()
    except Exception as error:
        return _chart_error(f"Unable to read the uploaded image: {error}", 400)
    if not content:
        return _chart_error("The uploaded image is empty.", 422)

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        return _chart_error(f"The uploaded file is not a readable image: {error}", 422)

    try:
        return analyze_chart_image(image, timeframe="10m")
    except Exception as error:
        # Keep the real internal error visible to the caller while preserving JSON.
        return _chart_error(f"Chart analysis failed internally: {error}", 500)
