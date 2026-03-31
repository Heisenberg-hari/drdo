from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

NUM_COLS = [
    "pulse_bpm",
    "temperature_c",
    "external_pressure",
    "body_pressure",
    "oximeter_reading",
    "motion_detection",
]

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)


def estimate_remaining_minutes(pulse: float, spo2: float, temp: float, motion: float, body_pressure: float) -> float:
    if pulse <= 0 or spo2 < 50:
        return 0.0

    score = (
        0.35 * (pulse / 72)
        + 0.35 * (spo2 / 98)
        + 0.15 * (temp / 36.8)
        + 0.10 * motion
        + 0.05 * (body_pressure / 85)
    )
    return max(0.0, score * 60)


def _clip_physically_impossible(df: pd.DataFrame) -> pd.DataFrame:
    df["pulse_bpm"] = df["pulse_bpm"].clip(20, 220)
    df["temperature_c"] = df["temperature_c"].clip(30.0, 45.0)
    df["oximeter_reading"] = df["oximeter_reading"].clip(30, 100)
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    motion_median = float(bundle.get("motion_median", 0.0))
    use_poly = bool(bundle.get("use_poly", False))

    out["pulse_temp_ratio"] = out["pulse_bpm"] / (out["temperature_c"] + 1e-6)
    out["pressure_diff"] = out["external_pressure"] - out["body_pressure"]
    out["oximeter_drop"] = 100 - out["oximeter_reading"]
    out["hypoxia_flag"] = (out["oximeter_reading"] < 90).astype(int)
    out["extreme_temp"] = ((out["temperature_c"] < 34) | (out["temperature_c"] > 38)).astype(int)
    out["high_motion"] = (out["motion_detection"] > motion_median).astype(int)

    if use_poly:
        for col in ["pulse_bpm", "oximeter_reading", "body_pressure", "temperature_c"]:
            out[f"{col}_sq"] = out[col] ** 2

    return out


def predict_from_payload(payload: dict) -> tuple[dict | None, str | None]:
    missing = [col for col in NUM_COLS if col not in payload]
    if missing:
        return None, f"Missing required fields: {missing}"

    try:
        sample = pd.DataFrame([[float(payload[col]) for col in NUM_COLS]], columns=NUM_COLS)
    except (TypeError, ValueError):
        return None, "All input fields must be numeric."

    sample[NUM_COLS] = bundle["imputer"].transform(sample[NUM_COLS])
    sample = _clip_physically_impossible(sample)
    sample = _engineer_features(sample)

    dead_probability = float(bundle["model"].predict_proba(sample[bundle["feature_cols"]])[:, 1][0])
    threshold = float(bundle["threshold"])
    status = "Dead" if dead_probability >= threshold else "Alive"

    response = {
        "status": status,
        "dead_probability": dead_probability,
        "threshold": threshold,
    }
    if status == "Alive":
        response["estimated_remaining_minutes"] = round(
            estimate_remaining_minutes(
                pulse=float(sample["pulse_bpm"].iloc[0]),
                spo2=float(sample["oximeter_reading"].iloc[0]),
                temp=float(sample["temperature_c"].iloc[0]),
                motion=float(sample["motion_detection"].iloc[0]),
                body_pressure=float(sample["body_pressure"].iloc[0]),
            ),
            2,
        )
    return response, None


@app.route("/", methods=["GET", "POST"])
def home():
    defaults = {
        "pulse_bpm": "72",
        "temperature_c": "36.8",
        "external_pressure": "101.3",
        "body_pressure": "85",
        "oximeter_reading": "98",
        "motion_detection": "0.2",
    }
    result = None
    error = None

    if request.method == "POST":
        payload = {k: request.form.get(k, "").strip() for k in NUM_COLS}
        defaults.update(payload)
        result, error = predict_from_payload(payload)

    return render_template_string(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 28px; background: #f4f6f8; }
    .card { max-width: 560px; background: #fff; padding: 18px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
    label { display: block; margin-top: 10px; font-size: 14px; }
    input { width: 100%; padding: 9px; margin-top: 4px; border: 1px solid #d1d5db; border-radius: 6px; }
    button { width: 100%; margin-top: 14px; padding: 10px; border: none; border-radius: 6px; background: #0f766e; color: #fff; }
    .res { margin-top: 14px; padding: 10px; background: #ecfeff; border-left: 4px solid #0f766e; }
    .err { margin-top: 12px; color: #b91c1c; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Prediction Form</h2>
    <form method="post">
      {% for field in fields %}
      <label for="{{ field }}">{{ field }}</label>
      <input id="{{ field }}" name="{{ field }}" value="{{ defaults[field] }}" required />
      {% endfor %}
      <button type="submit">Predict</button>
    </form>
    {% if error %}
      <div class="err">{{ error }}</div>
    {% endif %}
    {% if result %}
      <div class="res">
        <div><b>Status:</b> {{ result["status"] }}</div>
        <div><b>Dead Probability:</b> {{ "%.4f"|format(result["dead_probability"]) }}</div>
        <div><b>Threshold:</b> {{ "%.2f"|format(result["threshold"]) }}</div>
        {% if "estimated_remaining_minutes" in result %}
        <div><b>Estimated Remaining Minutes:</b> {{ result["estimated_remaining_minutes"] }}</div>
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
        """,
        fields=NUM_COLS,
        defaults=defaults,
        result=result,
        error=error,
    )


@app.post("/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    response, error = predict_from_payload(payload)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(response), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_path": str(MODEL_PATH), "model_loaded": True}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
