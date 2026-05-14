"""Improved infinite simulation — waits for ML training, tests all aspects."""
import urllib.request, json, time, random, sys

API = "http://localhost:8765"
TICKER = "AAPL"
BASE = 180.0

def call(path, data=None):
    if data:
        d = json.dumps(data).encode()
        req = urllib.request.Request(API+path, data=d, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(API+path)
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# 1. Train
print("=== Training %s ===" % TICKER, flush=True)
r = call("/api/train?ticker=" + TICKER)
print("Train:", r, flush=True)

# 2. Wait for training to complete (poll status)
print("=== Waiting for data fetch + ML training... ===", flush=True)
for _ in range(30):
    time.sleep(2)
    s = call("/api/status")
    if s.get("trained"):
        print("ML model trained! samples=%d" % s.get("samples", 0), flush=True)
        break
    trained_status = s.get("trained", False)
    samples = s.get("samples", 0)
    print("  status: trained=%s samples=%d" % (trained_status, samples), flush=True)
else:
    print("ML training not completed within 60s, continuing with rules", flush=True)

# 3. Send a few initial prices to build history naturally
for i in range(3):
    p = BASE + random.gauss(0, 0.3)
    r = call("/api/predict", {"ticker": TICKER, "price": round(p, 2)})
    time.sleep(1.0)

# 4. Infinite loop — simulate various market scenarios
phase = "neutral"
iteration = 0
price = BASE

while True:
    iteration += 1

    # Cycle through market phases every ~60 iterations
    if iteration % 60 == 0:
        phase = random.choice(["bull", "bear", "volatile", "neutral"])

    # Generate price based on phase
    if phase == "bull":
        drift = 0.3
        noise = 0.4
    elif phase == "bear":
        drift = -0.3
        noise = 0.4
    elif phase == "volatile":
        drift = random.gauss(0, 0.5)
        noise = 1.2
    else:  # neutral
        drift = random.gauss(0, 0.1)
        noise = 0.3

    change = drift + random.gauss(0, noise)
    price += change
    price = max(price, 0.5)
    price_rounded = round(price, 2)

    result = call("/api/predict", {"ticker": TICKER, "price": price_rounded})

    if "error" in result:
        print("[%4d] ERROR: %s" % (iteration, result["error"]), flush=True)
    else:
        pred = result.get("prediction", "?")
        conf = result.get("confidence", 0)
        trained = "ML" if result.get("trained") else "RULE"
        samples = result.get("samples", 0)
        ns = result.get("news_sentiment", 0)
        reasons = "; ".join(result.get("reasons", [])[:2])

        phase_icon = {"bull": "+", "bear": "-", "volatile": "~", "neutral": "."}[phase]

        print("[%4d] %s $%7.2f  %5s  %5.0f%%  [%s]  samples=%d  news=%+.2f  %s" % (
            iteration, phase_icon, price_rounded, pred, conf, trained, samples, ns, reasons), flush=True)

    time.sleep(1.0)
