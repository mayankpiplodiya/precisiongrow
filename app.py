# app.py (updated with ThingSpeak defaults + push endpoint)
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, Response
import numpy as np
import pickle
import traceback
from io import BytesIO
import os
import json
import urllib.request
import urllib.parse
import datetime
import logging

# Matplotlib non-interactive backend for servers without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'secretkey')

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrecisionGrow")

# ----------------- THINGSPEAK CONFIG (defaults + env override) -----------------
# Defaults set to the values you provided; override in production via env vars.
THINGSPEAK_CHANNEL_ID = os.environ.get("THINGSPEAK_CHANNEL_ID", "3139619")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY", "GKAG9JDF2CPJXWQD")   # used for reading feeds
THINGSPEAK_WRITE_KEY = os.environ.get("THINGSPEAK_WRITE_KEY", "GKAG9JDF2CPJXWQD") # used for writing (update)

logger.info("ThingSpeak channel=%s read_key_exists=%s write_key_exists=%s",
            THINGSPEAK_CHANNEL_ID,
            bool(THINGSPEAK_READ_KEY),
            bool(THINGSPEAK_WRITE_KEY))

# ----------------- LOAD MODEL AND SCALER -----------------
model = None
scaler = None
try:
    if os.path.exists('model.pkl'):
        model = pickle.load(open('model.pkl', 'rb'))
        logger.info("Model loaded from model.pkl (type=%s)", type(model))
    else:
        logger.warning("model.pkl not found — model is None")

    if os.path.exists('minmaxscaler.pkl'):
        scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
        logger.info("Scaler loaded from minmaxscaler.pkl (n_features_in_=%s)", getattr(scaler, 'n_features_in_', 'unknown'))
    else:
        logger.warning("minmaxscaler.pkl not found — scaler is None")
except Exception as e:
    logger.exception("Error loading model/scaler: %s", e)
    model = None
    scaler = None

# ----------------- USER AUTH -----------------
# Note: for demo only — in production, use a proper user store
users = {'admin': 'admin'}

# ----------------- CROP LABELS -----------------
label_map = {
    0: "rice", 1: "maize", 2: "chickpea", 3: "kidneybeans", 4: "pigeonpeas",
    5: "mothbeans", 6: "mungbean", 7: "blackgram", 8: "lentil", 9: "pomegranate",
    10: "banana", 11: "mango", 12: "grapes", 13: "watermelon", 14: "muskmelon",
    15: "apple", 16: "orange", 17: "papaya", 18: "coconut", 19: "cotton",
    20: "jute", 21: "coffee"
}

# ----------------- PDF LIBRARIES (optional) -----------------
FPDF_AVAILABLE = False
REPORTLAB_AVAILABLE = False
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    try:
        from reportlab.pdfgen import canvas
        REPORTLAB_AVAILABLE = True
    except Exception:
        logger.info("No PDF libraries available: falling back to text reports")

# ----------------- UTIL: ThingSpeak JSON fetch (no requests) -----------------
def fetch_thingspeak_json(channel_id=None, read_api_key=None, results=50):
    """
    Fetch ThingSpeak channel JSON feed using urllib (no external requests dependency).
    Returns parsed JSON dict or None.
    """
    try:
        channel = channel_id or THINGSPEAK_CHANNEL_ID
        api_key = read_api_key or THINGSPEAK_READ_KEY
        base = f"https://api.thingspeak.com/channels/{channel}/feeds.json?results={results}"
        if api_key:
            base += f"&api_key={api_key}"
        with urllib.request.urlopen(base, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw.decode('utf-8'))
    except Exception as e:
        logger.debug("fetch_thingspeak_json error: %s", e)
        return None

# ----------------- UTIL: ThingSpeak update (write) -----------------
def post_thingspeak(fields: dict, write_key=None):
    """
    Post a set of field values to ThingSpeak using the update API.
    fields: dict mapping 'field1'..'field8' or names like 'field1' -> value
    Returns dict { 'success': bool, 'response': str or code }
    """
    try:
        key = write_key or THINGSPEAK_WRITE_KEY
        if not key:
            return {"success": False, "response": "No write API key configured."}

        # prepare query params (only include numeric or string-converted values)
        params = {'api_key': key}
        # only take up to field1..field8 as ThingSpeak supports up to 8 fields
        for i in range(1, 9):
            fname = f'field{i}'
            if fname in fields and fields[fname] is not None:
                params[fname] = str(fields[fname])

        url = "https://api.thingspeak.com/update"
        data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp_text = resp.read().decode('utf-8')
            # ThingSpeak returns the entry id (int) on success, or '0'/'-1' on failure
            return {"success": True, "response": resp_text}
    except Exception as e:
        logger.exception("post_thingspeak error: %s", e)
        return {"success": False, "response": str(e)}

# ----------------- LIVE SENSOR DATA -----------------
def get_live_data():
    """
    Return the latest ThingSpeak feed mapped to expected keys.
    Adjust environment variables THINGSPEAK_CHANNEL_ID and THINGSPEAK_READ_KEY as needed.
    """
    try:
        data = fetch_thingspeak_json(THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_KEY, results=20)
        if not data or 'feeds' not in data or len(data['feeds']) == 0:
            return None, None

        feed = data['feeds'][-1]
        # map fields to expected sensor names (change if your mapping differs)
        live_data = {
            "N": float(feed.get('field1') or 0),
            "P": float(feed.get('field2') or 0),
            "K": float(feed.get('field3') or 0),
            "temperature": float(feed.get('field4') or 25),
            "humidity": float(feed.get('field5') or 50),
            "ph": float(feed.get('field6') or 7),
            "rainfall": float(feed.get('field7') or 0)
        }

        recommendations = {}
        if live_data['ph'] < 6:
            recommendations['pH'] = "Soil is acidic. Apply lime to balance pH."
        elif live_data['ph'] > 7.5:
            recommendations['pH'] = "Soil is alkaline. Apply elemental sulfur or organic matter to lower pH."

        if live_data['N'] < 50:
            recommendations['N'] = "Nitrogen is low. Apply urea or well-decomposed compost."
        if live_data['P'] < 30:
            recommendations['P'] = "Phosphorus is low. Apply DAP or bone meal."
        if live_data['K'] < 30:
            recommendations['K'] = "Potassium is low. Apply muriate of potash or wood ash."

        return live_data, recommendations
    except Exception as e:
        logger.exception("get_live_data error: %s", e)
        return None, None

# ----------------- ROUTES -----------------
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/crop-prediction')
def crop_prediction():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/soil-health')
def soil_health():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('soil_health.html')

@app.route('/reports')
def reports():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('reports.html')

# JSON endpoint used by Chart.js / front-end
@app.route('/soil-health-live')
def soil_health_live():
    data, rec = get_live_data()
    if not data:
        return jsonify({"error": "Could not fetch live data."}), 404
    return jsonify({"data": data, "recommendations": rec})

# Matplotlib PNG graph endpoint (optional)
@app.route('/live-graph')
def live_graph():
    """
    Returns a PNG image plotting the chosen ThingSpeak field.
    Query args:
      ?field=field4  (default)
      ?results=30
    """
    field = request.args.get('field', 'field4')
    results = int(request.args.get('results', 30))
    data = fetch_thingspeak_json(THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_KEY, results=results)
    if not data or 'feeds' not in data:
        return Response("No data", status=404)

    x = []
    y = []
    for f in data['feeds']:
        ts = f.get('created_at')
        val = f.get(field)
        try:
            if ts and val not in (None, ''):
                dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                x.append(dt)
                y.append(float(val))
        except Exception:
            continue

    if not y:
        return Response("No numeric data", status=404)

    plt.figure(figsize=(6, 3))
    plt.plot(x, y, marker='o', linewidth=1.2)
    plt.title(f"Live Feed — {field}")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# ----------------- PREDICT (robust) -----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        expected = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'pH']
        values = []
        missing = []
        invalid = []
        for name in expected:
            raw = request.form.get(name, None)
            if raw is None:
                missing.append(name)
                raw = ''
            raw = str(raw).strip()
            if raw == '':
                # treat blank as 0.0 (changeable)
                values.append(0.0)
            else:
                try:
                    values.append(float(raw))
                except ValueError:
                    invalid.append((name, raw))

        if missing:
            msg = f"Missing fields: {', '.join(missing)}"
            logger.warning("Predict - missing fields: %s", missing)
            return render_template('index.html', result=f"⚠️ {msg}")

        if invalid:
            pairs = ", ".join([f"{n}='{v}'" for n, v in invalid])
            logger.warning("Predict - invalid numeric input: %s", pairs)
            return render_template('index.html', result=f"⚠️ Invalid input: {pairs}")

        input_array = np.array(values).reshape(1, -1)
        logger.debug("Predict - raw input: %s", input_array.tolist())

        if scaler is not None:
            try:
                req = getattr(scaler, "n_features_in_", input_array.shape[1])
            except Exception:
                req = input_array.shape[1]

            if req > input_array.shape[1]:
                pad = np.zeros((1, req - input_array.shape[1]))
                input_array = np.hstack([input_array, pad])
            elif req < input_array.shape[1]:
                input_array = input_array[:, :req]

            try:
                scaled = scaler.transform(input_array)
            except Exception as e:
                logger.exception("Scaler transform error: %s", e)
                return render_template('index.html', result="⚠️ Scaler error — check scaler compatibility.")
        else:
            logger.warning("Predict - scaler is None, using raw inputs")
            scaled = input_array

        if model is None:
            logger.error("Predict - model not loaded")
            return render_template('index.html', result="⚠️ Model not loaded on server.")

        try:
            pred = model.predict(scaled)
            # normalize output to int index
            if isinstance(pred, (list, tuple, np.ndarray)):
                pred_val = int(pred[0])
            else:
                pred_val = int(pred)
        except Exception as e:
            logger.exception("Model predict error: %s", e)
            return render_template('index.html', result="⚠️ Model error during prediction.")

        predicted_label = label_map.get(pred_val, "Unknown Crop")
        logger.info("Prediction -> %s (%s)", pred_val, predicted_label)
        return render_template('index.html', result=f"🌾 Recommended Crop: {predicted_label}")

    except Exception as e:
        logger.exception("Unhandled /predict error: %s", e)
        return render_template('index.html', result="⚠️ Error processing input. See server logs.")

# ----------------- PUSH DATA TO THINGSPEAK (new) -----------------
@app.route('/push-sensor', methods=['POST'])
def push_sensor():
    """
    Accepts JSON or form data with keys field1..field8 (or N,P,K,temperature,humidity,ph,rainfall) and pushes them to ThingSpeak.
    Example JSON:
      { "field1": 12.3, "field2": 4.5 }
    Or:
      { "N": 12.3, "P": 4.5, "K": 7.8, "temperature": 25 }
    """
    if 'user' not in session:
        return jsonify({"success": False, "message": "Authentication required."}), 401

    payload = {}
    # try JSON first
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    # fallback to form data if JSON not provided
    if not payload:
        payload = request.form.to_dict()

    # Normalize known names to field1..field8
    mapping = {
        'N': 'field1', 'n': 'field1',
        'P': 'field2', 'p': 'field2',
        'K': 'field3', 'k': 'field3',
        'temperature': 'field4', 'temp': 'field4',
        'humidity': 'field5',
        'ph': 'field6',
        'rainfall': 'field7'
    }

    fields = {}
    for k, v in payload.items():
        if k in mapping:
            fields[mapping[k]] = v
        elif k.startswith('field') and k[5:].isdigit():
            fields[k] = v
        else:
            # ignore unknown keys but log
            logger.debug("push_sensor ignoring unknown key: %s", k)

    if not fields:
        return jsonify({"success": False, "message": "No valid fields provided."}), 400

    resp = post_thingspeak(fields, THINGSPEAK_WRITE_KEY)
    if resp.get("success"):
        return jsonify({"success": True, "response": resp.get("response")})
    else:
        return jsonify({"success": False, "response": resp.get("response")}), 500

# ----------------- REPORT (PDF/TXT) -----------------
@app.route('/download-soil-report')
def download_soil_report():
    data, recommendations = get_live_data()
    if not data:
        return "Live data unavailable. Cannot generate report.", 500

    try:
        if FPDF_AVAILABLE:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Soil Health Report", ln=True, align="C")
            pdf.ln(8)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, "Live Sensor Data:", ln=True)
            for k, v in data.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            pdf.ln(6)
            pdf.cell(0, 8, "Recommendations:", ln=True)
            for rec in recommendations.values():
                pdf.multi_cell(0, 8, f"- {rec}")
            buf = BytesIO()
            pdf.output(buf)
            buf.seek(0)
            return send_file(buf, as_attachment=True, download_name="soil_health_report.pdf", mimetype="application/pdf")
        elif REPORTLAB_AVAILABLE:
            from reportlab.pdfgen import canvas
            buf = BytesIO()
            c = canvas.Canvas(buf)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(300, 800, "Soil Health Report")
            c.setFont("Helvetica", 12)
            y = 760
            c.drawString(50, y, "Live Sensor Data:")
            y -= 20
            for k, v in data.items():
                c.drawString(60, y, f"{k}: {v}")
                y -= 16
            y -= 8
            c.drawString(50, y, "Recommendations:")
            y -= 20
            for rec in recommendations.values():
                c.drawString(60, y, f"- {rec}")
                y -= 16
            c.showPage()
            c.save()
            buf.seek(0)
            return send_file(buf, as_attachment=True, download_name="soil_health_report.pdf", mimetype="application/pdf")
    except Exception as e:
        logger.exception("PDF generation error: %s", e)

    # Fallback text report
    try:
        buf = BytesIO()
        lines = ["Soil Health Report\n", "Live Sensor Data:\n"]
        for k, v in data.items():
            lines.append(f"{k}: {v}\n")
        lines.append("\nRecommendations:\n")
        for r in recommendations.values():
            lines.append(f"- {r}\n")
        buf.write("".join(lines).encode('utf-8'))
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name="soil_health_report.txt", mimetype="text/plain")
    except Exception as e:
        logger.exception("Fallback report error: %s", e)
        return "Failed to generate report.", 500

# ----------------- CHATBOT -----------------
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message', '').lower().strip()
    except Exception:
        user_input = ''

    if not user_input:
        return jsonify({"reply": "👋 Ask me about soil pH, NPK, temp, humidity, or crops. Example: 'Ideal pH for rice?'"})

    general_responses = {
        "ph": "🧪 Ideal pH: 6.0–7.5. Lime for acidic, sulfur or organic matter for alkaline conditions.",
        "nitrogen": "🌿 Nitrogen deficiency → pale leaves. Apply urea or compost; use split doses to avoid burn.",
        "phosphorus": "🌱 Phosphorus supports roots & flowering. Apply DAP, bone meal or rock phosphate.",
        "potassium": "🥔 Potassium improves stress tolerance. Apply potash or wood ash."
    }

    crop_advice = {
        "rice": "Rice: prefers flooded/wet conditions, warm temps and balanced NPK.",
        "maize": "Maize: moderate humidity, 18–27°C, apply N in split doses for best uptake.",
        "banana": "Banana: high humidity & steady moisture, high K demand."
    }

    # simple keyword matching
    for k, resp in general_responses.items():
        if k in user_input:
            return jsonify({"reply": resp})

    for crop, info in crop_advice.items():
        if crop in user_input:
            return jsonify({"reply": f"🍃 For {crop.capitalize()}: {info}"})

    # dataset-driven hint if user asks "param for crop"
    if " for " in user_input or "for " in user_input:
        tokens = user_input.replace('?', '').split()
        param = None
        crop = None
        param_candidates = ['ph', 'n', 'p', 'k', 'temperature', 'temp', 'humidity', 'rainfall', 'nitrogen', 'phosphorus', 'potassium']
        for t in tokens:
            if t in param_candidates:
                param = t
            for v in label_map.values():
                if v in t:
                    crop = v
        if crop and param:
            try:
                import pandas as pd
                if os.path.exists("Crop_yield.csv"):
                    df = pd.read_csv("Crop_yield.csv")
                    col_map = {
                        'n': 'N', 'nitrogen': 'N',
                        'p': 'P', 'phosphorus': 'P',
                        'k': 'K', 'potassium': 'K',
                        'temperature': 'temperature', 'temp': 'temperature',
                        'humidity': 'humidity',
                        'ph': 'ph',
                        'rainfall': 'rainfall'
                    }
                    col = col_map.get(param, None)
                    if col and col in df.columns and 'label' in df.columns:
                        crop_df = df[df['label'].str.lower() == crop]
                        if not crop_df.empty:
                            mean_val = round(crop_df[col].mean(), 2)
                            return jsonify({"reply": f"📊 For {crop.capitalize()}, average {col} ≈ {mean_val}. Compare your reading to this."})
            except Exception:
                logger.debug("Chatbot dataset lookup failed", exc_info=True)

    return jsonify({"reply": "👋 I can help with soil pH, NPK, temp, humidity, and crop tips. Try: 'Ideal pH for rice?'"})

# ----------------- DEBUG INFO -----------------
@app.route('/debug-info')
def debug_info():
    """Return lightweight JSON about model/scaler status — only for dev use."""
    info = {
        "model_loaded": model is not None,
        "model_type": str(type(model)),
        "scaler_loaded": scaler is not None,
        "scaler_n_features_in": getattr(scaler, "n_features_in_", None),
        "env_thingspeak_channel": THINGSPEAK_CHANNEL_ID,
        # do not expose raw API keys; only indicate presence
        "env_thingspeak_read_key_exists": bool(THINGSPEAK_READ_KEY),
        "env_thingspeak_write_key_exists": bool(THINGSPEAK_WRITE_KEY)
    }
    return jsonify(info)

# ----------------- LOGOUT -----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ----------------- RUN APP -----------------
if __name__ == '__main__':
    # For production, run via gunicorn/uwsgi and set debug=False
    app.run(debug=True)
