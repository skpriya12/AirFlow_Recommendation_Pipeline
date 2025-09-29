import json
import time
from kafka import KafkaProducer


def validate_events(file_path: str):
    """
    Validate events.jsonl file and print any malformed lines.
    Returns a list of valid JSON objects.
    """
    valid_records = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                valid_records.append(record)
            except json.JSONDecodeError as e:
                print(f"❌ Line {i} is invalid JSON: {e}\n--> {line[:100]}...")
    print(f"✅ Found {len(valid_records)} valid records")
    return valid_records


def publish():
    """
    Publishes events from data/events.jsonl into the Kafka topic 'user_events'.
    Skips malformed JSON lines gracefully.
    """
    file_path = "/opt/airflow/data/events.jsonl"

    # Validate and collect good records
    events = validate_events(file_path)
    if not events:
        print("⚠️ No valid records found, nothing to publish")
        return

    # Configure Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=["kafka:9092"],  # container name if running in Docker
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    # Publish each event
    for record in events:
        producer.send("user_events", record)
        time.sleep(0.01)

    producer.flush()
    print(f"🚀 Published {len(events)} events to Kafka")


if __name__ == "__main__":
    publish()
