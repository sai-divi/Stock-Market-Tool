import urllib.request, json, time, random, sys

API = "http://localhost:8765"

def call(path, data=None):
    if data:
        d = json.dumps(data).encode()
        req = urllib.request.Request(API+path, data=d, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(API+path)
    r = urllib.request.urlopen(req, timeout=5)
    return json.loads(r.read())

# Train
print("Training...", flush=True)
r = call("/api/train?ticker=AAPL")
print(f"Train response: {r}", flush=True)
time.sleep(8)

# Send 10 simulated prices
base = 180.0
for i in range(10):
    base += random.gauss(0, 0.5)
    price = round(base, 2)
    r = call("/api/predict", {"ticker": "AAPL", "price": price})
    pred = r.get("prediction", "?")
    conf = r.get("confidence", 0)
    trained = r.get("trained", False)
    samples = r.get("samples", 0)
    reasons = r.get("reasons", [])
    print(f"[{i}] ${price:.2f} -> {pred}  conf={conf:.0f}%  trained={trained}  samples={samples}", flush=True)
    time.sleep(1)

print("Done. Check the app's Predict tab for live updates.", flush=True)
