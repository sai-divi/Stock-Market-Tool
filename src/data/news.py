import yfinance as yf
from datetime import datetime
from typing import List, Dict

_POSITIVE = {
    "surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies",
    "rallied", "jump", "jumps", "jumped", "gain", "gains", "gained", "rise",
    "rises", "rose", "rising", "up", "upgrade", "upgraded", "upgrades",
    "beat", "beats", "beaten", "positive", "strong", "stronger", "strongest",
    "growth", "growing", "profit", "profits", "profitable", "record", "high",
    "higher", "bullish", "buy", "outperform", "outperformed", "opportunity",
    "opportunities", "breakout", "launch", "launches", "launched", "boost",
    "boosts", "boosted", "momentum", "uptrend", "dividend", "buyback",
    "expansion", "innovate", "innovation", "partnership", "approval",
}

_NEGATIVE = {
    "drop", "drops", "dropped", "fall", "falls", "fell", "falling", "decline",
    "declines", "declined", "loss", "losses", "losing", "miss", "misses",
    "missed", "downgrade", "downgraded", "downgrades", "sell", "selling",
    "negative", "weak", "weaker", "weakness", "crash", "crashes", "crashed",
    "plunge", "plunges", "plunged", "low", "lower", "downturn", "bearish",
    "debt", "risk", "risky", "cut", "cuts", "cutting", "layoff", "layoffs",
    "fraud", "lawsuit", "investigation", "penalty", "fine", "sanction",
    "recession", "inflation", "volatile", "volatility", "warning", "downgrade",
    "underperform", "underperformed", "bear", "collapse", "bankruptcy",
    "default", "delist", "selloff", "sell-off", "outflow",
}


def _score_headline(title: str) -> float:
    words = set(title.lower().split())
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _extract(item: dict) -> dict:
    content = item.get("content") or item
    if not isinstance(content, dict):
        content = item
    title = content.get("title", "")
    if not title:
        title = content.get("headline", "")
    pub = content.get("pubDate") or content.get("publication_date") or ""
    if isinstance(pub, str) and pub:
        try:
            pub = datetime.strptime(pub.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").strftime("%m/%d %H:%M")
        except ValueError:
            pub = pub[:16]
    provider = content.get("provider", {})
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "") or provider.get("name", "")
    elif isinstance(provider, str):
        publisher = provider
    else:
        publisher = ""
    c_url = content.get("canonicalUrl", {})
    link = c_url.get("url", "") if isinstance(c_url, dict) else ""
    return {"title": title, "publisher": publisher, "date": pub, "link": link,
            "sentiment": _score_headline(title)}


def fetch_news(ticker: str, max_articles: int = 10) -> List[Dict]:
    try:
        tk = yf.Ticker(ticker)
        raw = tk.news or []
        articles = [_extract(item) for item in raw if isinstance(item, dict) and
                    (item.get("content") or item).get("title")]
        return articles[:max_articles]
    except Exception:
        return []


def aggregate_sentiment(articles: List[Dict]) -> Dict:
    if not articles:
        return {"avg": 0.0, "pos": 0, "neg": 0, "neutral": 0, "total": 0}
    scores = [a["sentiment"] for a in articles]
    return {
        "avg": round(sum(scores) / len(scores), 3),
        "pos": sum(1 for s in scores if s > 0.15),
        "neg": sum(1 for s in scores if s < -0.15),
        "neutral": sum(1 for s in scores if -0.15 <= s <= 0.15),
        "total": len(articles),
    }


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "BULLISH"
    if score < -0.15:
        return "BEARISH"
    return "NEUTRAL"
