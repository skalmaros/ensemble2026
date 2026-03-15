import os
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "task4"
API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")
NPZ_FILE = "submission.npz"


def main():
    if not os.path.exists(NPZ_FILE):
        raise FileNotFoundError(f"Brak pliku: {NPZ_FILE}")

    if not API_TOKEN:
        raise ValueError("TEAM_TOKEN not provided. Define TEAM_TOKEN in .env")

    if not SERVER_URL:
        raise ValueError("SERVER_URL not defined. Define SERVER_URL in .env")

    # Szybka weryfikacja przed wysłaniem
    data = np.load(NPZ_FILE)
    keys = list(data.keys())
    print(f"Wysyłam {NPZ_FILE}: {len(keys)} kluczy")
    print(f"Przykład: {keys[0]} -> {data[keys[0]].shape} {data[keys[0]].dtype}")
    data.close()

    headers = {"X-API-Token": API_TOKEN}

    response = requests.post(
        f"{SERVER_URL}/{ENDPOINT}",
        files={"npz_file": open(NPZ_FILE, "rb")},
        headers=headers
    )

    try:
        result = response.json()
    except Exception:
        result = response.text

    print("response:", response.status_code, result)


if __name__ == "__main__":
    main()