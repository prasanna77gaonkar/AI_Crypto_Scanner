"""Strict, conservative visual validation and analysis of candlestick charts."""

import logging

import cv2
import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


def _response(signal: str, confidence: int, trend: str, timeframe: str, analysis: str, message: str) -> dict:
    return {
        "status": "success",
        "message": message,
        "scanner": {
            "signal": signal,
            "confidence": max(0, min(100, int(confidence))),
            "trend": trend,
            "timeframe": timeframe,
            "analysis": analysis,
        },
    }


def _invalid_chart() -> dict:
    return _response(
        "INVALID CHART", 0, "UNKNOWN", "10m",
        "No reliable candlestick trading chart was detected.", "Image is not a valid trading chart",
    )


def _unclear_chart() -> dict:
    return _response(
        "HOLD", 0, "UNKNOWN", "10m",
        "A trading chart was detected, but the candlestick structure is not clear enough to analyze.",
        "Image analyzed",
    )


def _colour_masks(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coloured candle masks plus bright neutral/white candle evidence."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Use moderately saturated colours so photographed/blurred displays are not
    # discarded, while low-saturation text and most screen chrome stay excluded.
    bullish = cv2.inRange(hsv, np.array((35, 65, 55)), np.array((135, 255, 255)))
    bearish = cv2.bitwise_or(
        cv2.inRange(hsv, np.array((0, 65, 55)), np.array((20, 255, 255))),
        cv2.inRange(hsv, np.array((160, 65, 55)), np.array((180, 255, 255))),
    )
    # White candles are common on dark charts. Restrict neutral pixels to a
    # dark neighbourhood so a light page/chart background never becomes one
    # giant candidate component.
    bright_neutral = cv2.inRange(hsv, np.array((0, 0, 175)), np.array((180, 42, 255)))
    dark_pixels = cv2.inRange(hsv[:, :, 2], 0, 135)
    near_dark = cv2.dilate(dark_pixels, np.ones((9, 9), np.uint8))
    neutral = cv2.bitwise_and(bright_neutral, near_dark)
    return bullish, bearish, neutral


def _best_candle_run(candidates: list[dict]) -> list[dict]:
    """Keep the densest repeated-candle run and discard surrounding app UI."""
    if len(candidates) < 6:
        return []
    candidates.sort(key=lambda item: item["x"])
    widths = np.array([item["width"] for item in candidates], dtype=float)
    max_gap = max(20.0, float(np.median(widths)) * 7.0)
    runs: list[list[dict]] = []
    current = [candidates[0]]
    for candidate in candidates[1:]:
        if candidate["x"] - current[-1]["x"] <= max_gap:
            current.append(candidate)
        else:
            runs.append(current)
            current = [candidate]
    runs.append(current)
    eligible = [run for run in runs if len(run) >= 6]
    if not eligible:
        return []
    # Prefer repeated spacing over merely having many coloured UI components.
    def score(run: list[dict]) -> float:
        gaps = np.diff([item["x"] for item in run])
        spacing = 1.0 / (1.0 + float(np.std(gaps) / max(1.0, np.mean(gaps)))) if len(gaps) else 0.0
        return len(run) * (0.5 + spacing)
    return max(eligible, key=score)


def _chart_candidates(frame: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int] | None, float]:
    """Find repeated narrow coloured wick/body components; colour alone is insufficient."""
    height, width = frame.shape[:2]
    bullish, bearish, neutral = _colour_masks(frame)
    combined = cv2.bitwise_or(cv2.bitwise_or(bullish, bearish), neutral)
    # Join a body to its vertical wick, but never join neighbouring candles.
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    candidates: list[dict] = []
    geometry_passes = 0
    for label in range(1, component_count):
        x, y, candidate_width, candidate_height, area = (int(value) for value in stats[label])
        if not (2 <= candidate_width <= max(12, width // 18)):
            continue
        if not (4 <= candidate_height <= int(height * 0.80)):
            continue
        fill = area / max(1, candidate_width * candidate_height)
        aspect = candidate_height / candidate_width
        # A candle is a narrow vertical wick/body, not a large coloured object.
        if aspect < 1.15 or fill < 0.05 or fill > 0.98:
            continue
        component = combined[y:y + candidate_height, x:x + candidate_width]
        row_widths = np.count_nonzero(component, axis=1)
        max_row_width = int(row_widths.max())
        if max_row_width < 2:
            continue
        # A candle must have a multi-row wide body and narrower pixels extending
        # above and/or below it as a wick. This rejects coloured poles, text,
        # repeated rectangles, and ordinary photo objects.
        body_rows = np.flatnonzero(row_widths >= max(2, int(max_row_width * 0.68)))
        if body_rows.size < 2:
            continue
        body_top, body_bottom = int(body_rows[0]), int(body_rows[-1])
        body_height = body_bottom - body_top + 1
        thin_limit = max(2, int(max_row_width * 0.55))
        upper_wick = int(np.count_nonzero((row_widths[:body_top] > 0) & (row_widths[:body_top] <= thin_limit)))
        lower_wick = int(np.count_nonzero((row_widths[body_bottom + 1:] > 0) & (row_widths[body_bottom + 1:] <= thin_limit)))
        # Wicks may be invisible in a blurred or partially occluded photo;
        # repeated rectangular bodies remain valid chart evidence.
        if body_height < 2:
            continue
        geometry_passes += 1
        crop_green = bullish[y:y + candidate_height, x:x + candidate_width]
        crop_red = bearish[y:y + candidate_height, x:x + candidate_width]
        crop_neutral = neutral[y:y + candidate_height, x:x + candidate_width]
        green_pixels = int(np.count_nonzero(crop_green))
        red_pixels = int(np.count_nonzero(crop_red))
        neutral_pixels = int(np.count_nonzero(crop_neutral))
        if max(green_pixels, red_pixels) < max(3, neutral_pixels * 0.25):
            direction = 0
        else:
            direction = 1 if green_pixels >= red_pixels else -1
        absolute_body_top = y + body_top
        absolute_body_bottom = y + body_bottom
        candidates.append({
            "x": x + candidate_width / 2,
            "y": y + candidate_height / 2,
            "width": candidate_width,
            "height": candidate_height,
            "body_height": body_height,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "direction": direction,
            "fill": fill,
            # Image y grows downward. These are pixel-space OHLC values and
            # are converted to price direction only during analysis.
            "high": y,
            "low": y + candidate_height - 1,
            "open": absolute_body_bottom if direction > 0 else absolute_body_top,
            "close": absolute_body_top if direction > 0 else absolute_body_bottom,
        })

    candidates = _best_candle_run(candidates)
    if len(candidates) < 6:
        LOGGER.debug("chart validation rejected: candidate_count=%s geometry_passes=%s", len(candidates), geometry_passes)
        return candidates, None, 0.0
    candidates.sort(key=lambda item: item["x"])
    xs = np.array([item["x"] for item in candidates], dtype=float)
    ys = np.array([item["y"] for item in candidates], dtype=float)
    left, right = int(xs.min()), int(xs.max())
    top = max(0, int(min(item["y"] - item["height"] / 2 for item in candidates) - 10))
    bottom = min(height, int(max(item["y"] + item["height"] / 2 for item in candidates) + 10))
    coverage = (right - left) / max(1.0, width)
    gaps = np.diff(xs)
    positive_gaps = gaps[gaps > 1]
    spacing = 0.0
    if positive_gaps.size >= 6:
        spacing = 1.0 / (1.0 + float(np.std(positive_gaps) / max(1.0, np.mean(positive_gaps))))
    vertical_spread = (float(ys.max()) - float(ys.min())) / max(1.0, bottom - top)
    heights = np.array([item["height"] for item in candidates], dtype=float)
    bodies = np.array([item["body_height"] for item in candidates], dtype=float)
    size_consistency = 1.0 / (1.0 + float(np.std(heights) / max(1.0, np.mean(heights))))
    body_consistency = 1.0 / (1.0 + float(np.std(bodies) / max(1.0, np.mean(bodies))))
    wick_balance = float(np.mean([
        min(1.0, (item["upper_wick"] + item["lower_wick"]) / max(1.0, item["body_height"]))
        for item in candidates
    ]))
    wick_candidate_ratio = float(np.mean([
        item["upper_wick"] + item["lower_wick"] >= 2
        for item in candidates
    ]))
    two_sided_wick_ratio = float(np.mean([
        item["upper_wick"] > 0 and item["lower_wick"] > 0
        for item in candidates
    ]))
    chart_area_ratio = ((right - left) * (bottom - top)) / max(1.0, width * height)
    sequence_aspect = (right - left) / max(1.0, bottom - top)

    # Plot regions normally contain horizontal/vertical grid or border edges.
    roi = cv2.cvtColor(frame[top:bottom, max(0, left - 25):min(width, right + 25)], cv2.COLOR_BGR2GRAY)
    if roi.size == 0:
        return candidates, None, 0.0
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 45, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=22, minLineLength=max(20, roi.shape[1] // 8), maxLineGap=8)
    line_support = 0.0
    if lines is not None:
        segments = np.asarray(lines).reshape(-1, 4)  # Handles OpenCV's (N, 1, 4) safely.
        horizontal_or_vertical = 0
        for x1, y1, x2, y2 in segments:
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            if dx >= 3 * max(1, dy) or dy >= 3 * max(1, dx):
                horizontal_or_vertical += 1
        line_support = min(1.0, horizontal_or_vertical / 8)

    count_score = min(1.0, len(candidates) / 18)
    geometry_score = min(1.0, float(np.median([item["height"] / item["width"] for item in candidates])) / 3)
    repeated_pattern = 0.35 * spacing + 0.25 * size_consistency + 0.25 * body_consistency + 0.15 * wick_balance
    validation = (
        0.20 * count_score + 0.17 * coverage + 0.18 * geometry_score
        + 0.25 * repeated_pattern + 0.10 * min(1.0, chart_area_ratio * 5) + 0.10 * line_support
    )
    # Validation is evidence-based rather than a single rigid crop/size rule.
    # A readable image, colours, text, rectangles, or vertical edges cannot
    # pass alone: several independent candle-sequence characteristics must
    # agree. This accepts cropped/light/mobile charts with fewer candles while
    # continuing to reject ordinary photographs and unrelated screenshots.
    independent_signals = sum((
        len(candidates) >= 6,
        coverage >= 0.10,
        spacing >= 0.38,
        geometry_score >= 0.40,
        size_consistency >= 0.30 and body_consistency >= 0.28,
        wick_candidate_ratio >= 0.40 or two_sided_wick_ratio >= 0.18,
        sequence_aspect >= 1.25,
        chart_area_ratio >= 0.008 or line_support >= 0.20,
    ))
    valid = (
        len(candidates) >= 6 and wick_candidate_ratio >= 0.34
        and spacing >= 0.32 and sequence_aspect >= 1.10
        and repeated_pattern >= 0.34 and validation >= 0.40
        and independent_signals >= 6
    )
    LOGGER.debug(
        "chart validation metrics count=%s coverage=%.3f spacing=%.3f geometry=%.3f size=%.3f body=%.3f "
        "wicks=%.3f wick_candidates=%.3f two_sided=%.3f area=%.3f aspect=%.3f pattern=%.3f lines=%.3f score=%.3f signals=%s valid=%s",
        len(candidates), coverage, spacing, geometry_score, size_consistency, body_consistency,
        wick_balance, wick_candidate_ratio, two_sided_wick_ratio, chart_area_ratio, sequence_aspect,
        repeated_pattern, line_support, validation, independent_signals, valid,
    )
    return candidates, (left, top, right, bottom) if valid else None, validation


def _quality(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 180.0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    glare = float(np.mean((hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 235)))
    return sharpness, glare


def _price(candle: dict, key: str) -> float:
    """Convert screen coordinates to price-like coordinates (up is positive)."""
    return -float(candle[key])


def _candle_pattern(candles: list[dict]) -> tuple[int, str]:
    """Return a conservative pattern score and a description for the latest candle."""
    latest = candles[-1]
    candle_range = max(1.0, _price(latest, "high") - _price(latest, "low"))
    body = abs(_price(latest, "close") - _price(latest, "open"))
    upper = _price(latest, "high") - max(_price(latest, "open"), _price(latest, "close"))
    lower = min(_price(latest, "open"), _price(latest, "close")) - _price(latest, "low")
    bullish = latest["direction"] > 0

    if len(candles) >= 2:
        previous = candles[-2]
        previous_bullish = previous["direction"] > 0
        if bullish and not previous_bullish and _price(latest, "open") <= _price(previous, "close") and _price(latest, "close") >= _price(previous, "open"):
            return 10, "bullish engulfing"
        if not bullish and previous_bullish and _price(latest, "open") >= _price(previous, "close") and _price(latest, "close") <= _price(previous, "open"):
            return -10, "bearish engulfing"
    if body / candle_range <= 0.12:
        return 0, "doji"
    if lower >= body * 2 and upper <= max(1.0, body * 0.75):
        return 8, "hammer"
    if upper >= body * 2 and lower <= max(1.0, body * 0.75):
        return -8, "shooting star"
    if body / candle_range >= 0.65:
        return (6 if bullish else -6), ("strong bullish candle" if bullish else "strong bearish candle")
    return 0, "no clear reversal pattern"


def _insufficient_chart() -> dict:
    return _response(
        "HOLD", 20, "UNKNOWN", "10m",
        "The chart is valid, but there is insufficient evidence for a directional signal.",
        "10-minute chart analyzed successfully",
    )


def analyze_chart_image(image: Image.Image, timeframe: str = "10m") -> dict:
    """Validate first; only valid, clear candlestick charts reach signal analysis."""
    if timeframe != "10m":
        return _invalid_chart()
    try:
        frame = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        if width < 160 or height < 120:
            return _invalid_chart()

        candidates, region, validation_score = _chart_candidates(frame)
        if region is None:
            return _invalid_chart()

        sharpness, glare = _quality(frame)
        # A camera image may be soft or reflective; those lower confidence but
        # should not erase otherwise repeated candle evidence.
        if sharpness < 0.025 or glare > 0.62:
            return _unclear_chart()

        # Candles are ordered by x, so this is deterministic oldest-to-newest
        # analysis of the visible chart rather than a random image label.
        if len(candidates) < 10:
            return _insufficient_chart()
        recent = candidates[-min(10, len(candidates)):]
        closes = np.array([_price(item, "close") for item in recent])
        highs = np.array([_price(item, "high") for item in recent])
        lows = np.array([_price(item, "low") for item in recent])
        directions = np.array([item["direction"] for item in recent], dtype=float)
        ranges = np.maximum(1.0, highs - lows)
        typical_range = float(np.median(ranges))
        x_positions = np.arange(len(recent), dtype=float)
        close_slope = float(np.polyfit(x_positions, closes, 1)[0])
        movement = (float(np.mean(closes[-3:])) - float(np.mean(closes[:3]))) / typical_range
        higher_highs = float(np.mean(np.diff(highs) > typical_range * 0.08))
        higher_lows = float(np.mean(np.diff(lows) > typical_range * 0.08))
        bullish_ratio = float(np.mean(directions > 0))
        bearish_ratio = 1.0 - bullish_ratio
        structure = (higher_highs + higher_lows) / 2

        score = 0
        reasons: list[str] = []
        normalized_slope = close_slope / typical_range
        recent_weights = np.linspace(0.55, 1.0, len(recent))
        recent_bias = float(np.dot(directions, recent_weights) / np.sum(recent_weights))
        signed_body_strength = float(np.mean([
            item["direction"] * abs(_price(item, "close") - _price(item, "open")) / max(1.0, _price(item, "high") - _price(item, "low"))
            for item in recent[-4:]
        ]))

        # Trend combines higher-high/lower-low structure with the close slope.
        # Moderate agreement earns evidence; strong agreement earns more.
        if structure >= 0.50 and normalized_slope >= 0.05:
            trend_points = 25 if structure >= 0.67 and normalized_slope >= 0.11 else 15
            score += trend_points
            reasons.append("higher highs and higher lows")
        elif structure <= 0.50 and normalized_slope <= -0.05:
            trend_points = 25 if structure <= 0.33 and normalized_slope <= -0.11 else 15
            score -= trend_points
            reasons.append("lower highs and lower lows")

        # Momentum compares the final three candles with the initial three.
        if movement >= 0.85:
            score += 25
            reasons.append("strong positive momentum")
        elif movement >= 0.30:
            score += 10
            reasons.append("positive momentum")
        elif movement <= -0.85:
            score -= 25
            reasons.append("strong negative momentum")
        elif movement <= -0.30:
            score -= 10
            reasons.append("negative momentum")

        if bullish_ratio >= 0.72:
            score += 20
            reasons.append("repeated bullish candles")
        elif bullish_ratio >= 0.58:
            score += 12
            reasons.append("bullish candle majority")
        elif bearish_ratio >= 0.72:
            score -= 20
            reasons.append("repeated bearish candles")
        elif bearish_ratio >= 0.58:
            score -= 12
            reasons.append("bearish candle majority")

        # Recent candles carry more weight than the older visible history.
        if recent_bias >= 0.55 and signed_body_strength >= 0.25:
            score += 15
            reasons.append("strong recent bullish candles")
        elif recent_bias >= 0.22:
            score += 8
            reasons.append("recent bullish candle strength")
        elif recent_bias <= -0.55 and signed_body_strength <= -0.25:
            score -= 15
            reasons.append("strong recent bearish candles")
        elif recent_bias <= -0.22:
            score -= 8
            reasons.append("recent bearish candle strength")

        pattern_score, pattern_name = _candle_pattern(recent)
        score += pattern_score
        if pattern_score:
            reasons.append(pattern_name)

        latest_close = closes[-1]
        support = float(np.percentile(lows, 20))
        resistance = float(np.percentile(highs, 80))
        near_support = latest_close - support <= typical_range * 0.65
        near_resistance = resistance - latest_close <= typical_range * 0.65
        if near_support and (bullish_ratio >= 0.55 or pattern_score > 0):
            score += 15
            reasons.append("bullish evidence near visible support")
        elif near_resistance and (bearish_ratio >= 0.55 or pattern_score < 0):
            score -= 15
            reasons.append("bearish evidence near visible resistance")

        breakout_up = latest_close >= float(np.percentile(highs, 80)) - typical_range * 0.30
        breakout_down = latest_close <= float(np.percentile(lows, 20)) + typical_range * 0.30
        if breakout_up and movement > 0 and recent_bias > 0:
            score += 10
            reasons.append("upward breakout confirmation")
        elif breakout_down and movement < 0 and recent_bias < 0:
            score -= 10
            reasons.append("downward breakdown confirmation")

        score = max(-100, min(100, score))
        evidence_quality = min(1.0, 0.55 * validation_score + 0.25 * min(1.0, len(recent) / 10) + 0.20 * max(bullish_ratio, bearish_ratio))
        confidence = min(95, round((15 + abs(score) * 0.68 + evidence_quality * 22) * max(0.55, sharpness) * (1.0 - min(0.45, glare))))

        if score >= 55:
            signal, trend = "BUY", "BULLISH"
        elif score <= -55:
            signal, trend = "SELL", "BEARISH"
        else:
            signal = "HOLD"
            trend = "BULLISH" if score >= 20 else "BEARISH" if score <= -20 else "SIDEWAYS"
            confidence = min(confidence, 54)
        if signal == "HOLD":
            analysis = (
                "Recent price movement is mixed and does not provide sufficient directional confirmation."
                if trend == "SIDEWAYS"
                else f"Recent {trend.lower()} evidence is present, but the detected score ({score:+d}) is below the directional-signal threshold."
            )
        else:
            analysis = f"Recent candles show {', '.join(reasons[:3])}. Signal is based on detected technical evidence only, not a profit guarantee."
        return _response(signal, confidence, trend, "10m", analysis, "10-minute chart analyzed successfully")
    except (cv2.error, ValueError, TypeError, np.linalg.LinAlgError):
        return _invalid_chart()
