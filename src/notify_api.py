import requests

def notify(**context):
    """
    Notifies the reco API to trigger model training.
    """
    url = "http://reco_api:8000/train"

    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        print(f"✅ Notified reco API: {response.status_code} {response.text}")
    except requests.RequestException as e:
        print(f"❌ Failed to notify reco API: {e}")
        raise
