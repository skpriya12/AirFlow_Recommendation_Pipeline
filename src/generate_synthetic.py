import json, random, uuid
from datetime import datetime, timedelta
from pathlib import Path

CATEGORIES = ["electronics", "apparel", "home", "beauty"]
DEVICES = ["web", "ios", "android"]


def make_event(ts, user_id, session_id):
    return {
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,  # 👈 new field
        "event_ts": ts.isoformat(),
        "user_id": user_id,
        "event_type": random.choice(["view", "add_to_cart", "purchase"]),
        "product_id": str(uuid.uuid4())[:8],
        "category": random.choice(CATEGORIES),
        "price": round(random.uniform(5, 500), 2),
        "device": random.choice(DEVICES),
    }


def generate(n_users=500, days=1, out_path="data/events.jsonl"):
    Path("data").mkdir(exist_ok=True)
    now = datetime.utcnow()

    with open(out_path, "w") as f:
        for u in range(n_users):
            user = f"u_{u:03d}"
            base = now - timedelta(days=days)

            # Each user has multiple sessions
            for s in range(random.randint(1, 3)):  # 1–3 sessions per user
                session_id = str(uuid.uuid4())
                for _ in range(random.randint(5, 15)):  # events per session
                    ts = base + timedelta(seconds=random.randint(0, days * 86400))
                    f.write(json.dumps(make_event(ts, user, session_id)) + "\n")

    print(f"✅ Saved synthetic events with session_id to {out_path}")


if __name__ == "__main__":
    generate()
