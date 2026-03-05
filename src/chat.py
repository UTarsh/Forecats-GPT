"""
Natural-language layer over the demand forecasting model.

No external AI API required — this module uses regex to pull dates, prices,
promos, and stockout flags from free-text questions, runs the model, and
writes a plain-English explanation of the result.
"""

import datetime
import re

try:
    from src.predict import predict_one
except ImportError:
    from predict import predict_one

# ── Keywords that signal the user wants a prediction ─────────────────────────
PREDICTION_KEYWORDS = [
    "predict", "forecast", "demand", "units", "sales", "how many",
    "what if", "what will", "will sell", "sell tomorrow", "sell today",
    "expect", "projection", "estimate",
]

HELP_MESSAGE = """Hi! I'm your **demand forecasting assistant**.

Ask me things like:

- *"Predict demand for tomorrow with a promo at $18"*
- *"What will sales be if price is $22 and competitor is $21?"*
- *"How many units if we're out of stock next Monday?"*
- *"What if we run a promotion today at $19.99?"*

I'll run the forecasting model and explain the results in plain English!"""


# ── Date parser ───────────────────────────────────────────────────────────────

def _parse_date(text: str) -> str:
    today = datetime.date.today()
    t = text.lower()

    # Explicit YYYY-MM-DD
    m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
    if m:
        return m.group(1)

    if "tomorrow" in t:
        return str(today + datetime.timedelta(days=1))

    if "today" in t or "tonight" in t:
        return str(today)

    if "next week" in t:
        return str(today + datetime.timedelta(days=7))

    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, day_num in day_map.items():
        if day_name in t:
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return str(today + datetime.timedelta(days=days_ahead))

    return str(today)  # default to today


# ── Price parser ──────────────────────────────────────────────────────────────

def _parse_prices(text: str) -> tuple:
    """Returns (our_price, competitor_price) — None means not found."""
    t = text.lower()
    our_price = None
    comp_price = None

    # Competitor price (look for explicit label first)
    m = re.search(r'(?:competitor|comp)[^\d$]*\$?(\d+\.?\d*)', t)
    if m:
        comp_price = float(m.group(1))

    # Our price — try several patterns in priority order
    for pattern in [
        r'(?:our\s+price|my\s+price|price\s+is|priced\s+at)\s+\$?(\d+\.?\d*)',
        r'(?:at|for)\s+\$(\d+\.?\d*)',
        r'price[:\s]+\$?(\d+\.?\d*)',
    ]:
        m = re.search(pattern, t)
        if m:
            val = float(m.group(1))
            # Don't steal the competitor price
            if comp_price is None or abs(val - comp_price) > 0.01:
                our_price = val
                break

    # Fallback: lone $XX.XX in the message
    if our_price is None and comp_price is None:
        m = re.search(r'\$(\d+\.?\d*)', text)
        if m:
            our_price = float(m.group(1))

    return our_price, comp_price


# ── Full message parser ───────────────────────────────────────────────────────

def parse_message(text: str) -> dict:
    """Extract prediction parameters from a natural language message."""
    t = text.lower()

    date_str = _parse_date(text)
    our_price, comp_price = _parse_prices(text)

    price_assumed = our_price is None
    comp_assumed = comp_price is None

    if our_price is None:
        our_price = 20.0
    if comp_price is None:
        comp_price = round(our_price + 0.40, 2)

    promo_words = ["promo", "promotion", "discount", "sale", "deal", "offer", "markdown"]
    promo = 1 if any(w in t for w in promo_words) else 0

    stockout_words = ["stockout", "out of stock", "no stock", "unavailable", "sold out", "stock out"]
    stockout = 1 if any(w in t for w in stockout_words) else 0

    return {
        "date": date_str,
        "price": round(our_price, 2),
        "competitor_price": round(comp_price, 2),
        "promo": promo,
        "stockout": stockout,
        "_price_assumed": price_assumed,
        "_comp_assumed": comp_assumed,
    }


# ── Response generator ────────────────────────────────────────────────────────

def _format_date(date_str: str) -> str:
    d = datetime.date.fromisoformat(date_str)
    return d.strftime("%A, %B") + f" {d.day}"


def generate_response(params: dict, prediction: float) -> str:
    day_str = _format_date(params["date"])
    stockout_tag = ", **OUT OF STOCK**" if params["stockout"] else ""

    # Build inputs summary
    price_label = f"price **${params['price']:.2f}**" + (" *(assumed)*" if params["_price_assumed"] else "")
    comp_label = f"competitor **${params['competitor_price']:.2f}**" + (" *(assumed)*" if params["_comp_assumed"] else "")
    promo_label = "**with promotion**" if params["promo"] else "no promotion"

    lines = [
        f"**Forecast for {day_str}**{stockout_tag}",
        "",
        f"Inputs: {price_label} | {comp_label} | {promo_label}",
        "",
        f"### Predicted demand: **{round(prediction)} units**",
        "",
    ]

    # Add contextual insights
    if params["stockout"]:
        lines.append("⚠️ **Stockout alert** — being out of stock reduces demand by ~85 units. Restocking should recover most of those sales.")
    if params["promo"]:
        lines.append("🎯 **Promotion boost** — the promo adds approximately +30 units of demand.")
    if params["competitor_price"] < params["price"]:
        diff = params["price"] - params["competitor_price"]
        lines.append(f"📉 **Price gap** — competitor is ${diff:.2f} cheaper, which is pulling demand away from you.")
    elif params["competitor_price"] > params["price"] + 1.0:
        lines.append("✅ **Price advantage** — you're cheaper than the competitor, which helps demand.")
    if params["price"] < 19.0:
        lines.append("💲 **Low price** — your below-average price is boosting demand.")
    elif params["price"] > 21.5:
        lines.append("💲 **High price** — your above-average price is dampening demand slightly.")

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def process_message(message: str) -> str:
    """Turn a natural language message into a prediction + plain-English reply."""
    if not any(kw in message.lower() for kw in PREDICTION_KEYWORDS):
        return HELP_MESSAGE

    try:
        params = parse_message(message)
        prediction = predict_one(params)
        return generate_response(params, prediction)
    except FileNotFoundError:
        return "⚠️ Model not found. Please run `python src/train.py` first, then restart the server."
    except Exception as e:
        return f"Sorry, I ran into an issue: `{e}`"
