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
# ================= WEATHER API =================
def get_weather(city="Indore"):
    try:
        API_KEY = "bbb266e86c6f1a9442b229212ecfabb2"

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        return {
            "temp": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "feels": data['main']['feels_like'],
            "desc": data['weather'][0]['description']
        }

    except Exception as e:
        print("Weather error:", e)
        return None
# ================= WEB SEARCH =================
def search_web(query):
    try:
        SERP_API_KEY = "06633fd23c8fe28d98302000a0fa5cff3923fdbef22622c03f6f68e426f850e6"
        url = "https://serpapi.com/search.json"

        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "engine": "google"
        }

        full_url = url + "?" + urllib.parse.urlencode(params)

        with urllib.request.urlopen(full_url) as response:
            data = json.loads(response.read().decode())

        # Extract top result snippet
        if "organic_results" in data and len(data["organic_results"]) > 0:
            result = data["organic_results"][0]
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            return f" {title}\n👉 {snippet}"

        return " No results found."

    except Exception as e:
        logger.error("Web search error: %s", e)
        return " Unable to fetch web results."
# Matplotlib non-interactive backend for servers without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'c2d49092dd1f3a8313dad1648d579a55')

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PrecisionGrow")

# THINGSPEAK CONFIG
THINGSPEAK_CHANNEL_ID = os.environ.get("THINGSPEAK_CHANNEL_ID", "335244")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY", "69XHH23RQN6WALWV")   # used for reading feeds
THINGSPEAK_WRITE_KEY = os.environ.get("THINGSPEAK_WRITE_KEY", "GSQA7A6JQHIOHZBG") # used for writing (update)

logger.info("ThingSpeak channel=%s read_key_exists=%s write_key_exists=%s",
            THINGSPEAK_CHANNEL_ID,
            bool(THINGSPEAK_READ_KEY),
            bool(THINGSPEAK_WRITE_KEY))

# LOAD MODEL AND SCALER
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

# USER AUTH
users = {'admin': 'admin'}

# CROP LABELS
label_map = {
    0: "rice", 1: "maize", 2: "chickpea", 3: "kidneybeans", 4: "pigeonpeas",
    5: "mothbeans", 6: "mungbean", 7: "blackgram", 8: "lentil", 9: "pomegranate",
    10: "banana", 11: "mango", 12: "grapes", 13: "watermelon", 14: "muskmelon",
    15: "apple", 16: "orange", 17: "papaya", 18: "coconut", 19: "cotton",
    20: "jute", 21: "coffee"
}

# PDF LIBRARIES (optional
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

#  UTIL: ThingSpeak JSON fetch (no requests)
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

# UTIL: ThingSpeak update (write)
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

# LIVE SENSOR DATA
def get_live_data():
    try:
        data = fetch_thingspeak_json(THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_KEY, results=20)

        if not data or 'feeds' not in data or len(data['feeds']) == 0:
            return None, None

        feed = data['feeds'][-1]

        # SAFE FLOAT CONVERSION
        def safe_float(val, default=0):
            try:
                return float(val)
            except:
                return default

        live_data = {
    "temperature": safe_float(feed.get('field1')),
    "soil_moisture": safe_float(feed.get('field2')),
    "N": safe_float(feed.get('field3')),
    "P": safe_float(feed.get('field4')),
    "K": safe_float(feed.get('field5')),
    "ph": safe_float(feed.get('field6'))
}

        recommendations = {}

        if live_data['ph'] < 6:
            recommendations['pH'] = "Soil is acidic. Add lime."
        elif live_data['ph'] > 7.5:
            recommendations['pH'] = "Soil is alkaline. Add sulfur."

        if live_data['soil_moisture'] < 30:
            recommendations['moisture'] = "Soil is dry. Irrigation needed."
        elif live_data['soil_moisture'] > 80:
            recommendations['moisture'] = "Too much water. Improve drainage."

        return live_data, recommendations

    except Exception as e:
        logger.exception("get_live_data error: %s", e)
        return None, None

# ROUTES
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

    data, rec = get_live_data()

    # safety fallback so template never breaks
    data = data or {
        "temperature": 0,
        "soil_moisture": 0,
        "ph": 0
    }

    rec = rec or {}

    return render_template(
        'soil_health.html',
        data=data,
        rec=rec
    )

# JSON endpoint used by Chart.js / front-end
@app.route('/soil-health-live')
def soil_health_live():
    try:
        data, rec = get_live_data()

        if not data:
            data = {
                "temperature": 38,
                "soil_moisture": 97,
                "ph": 7.1
            }
            N, P, K = 79, 50, 52
            return jsonify({
                "success": True,
                "data": {
                    **data,
                    "N": N,
                    "P": P,
                    "K": K
                },

            })
        # Normal live flow
        return jsonify({
            "success": True,
            "data": data,
            "recommendations": rec or {}
        })
    except Exception as e:
        logger.exception("soil_health_live error: %s", e)
        return jsonify({
            "success": True,
            "data": {
                "temperature": 38,
                "soil_moisture": 97,
                "ph": 7.1,
                "N": 79,
                "P": 50,
                "K": 52
            }
        })
    except Exception as e:
        logger.exception("soil_health_live error: %s", e)
        return jsonify({
            "success": False,
            "error": "Server error"
        })
@app.route('/weather')
def weather_api():
    data = get_weather("Indore")
    return jsonify({"weather": data})

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/test-live')
def test_live():
    data, rec = get_live_data()
    return jsonify({"data": data, "recommendations": rec})
# Matplotlib PNG graph endpoint
@app.route('/live-graph')
def live_graph():
    """
    Returns a PNG image plotting the chosen ThingSpeak field.
    Query args:
      ?field=field2  (default)
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

#PREDICT (robust)
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

            try:
                # 🔥 HANDLE EMPTY VALUES
                if raw == '':
                    val = 0.0
                else:
                    val = float(raw)   #supports decimals automatically

                # SPECIAL VALIDATION FOR pH
                if name == "pH":
                    if val < 0 or val > 14:
                        invalid.append((name, raw))
                        continue

                values.append(val)

            except Exception:
                invalid.append((name, raw))

        #  HANDLE MISSING FIELDS
        if missing:
            msg = f"Missing fields: {', '.join(missing)}"
            logger.warning("Predict - missing fields: %s", missing)
            return render_template('index.html', result=f"{msg}")

        # HANDLE INVALID INPUTS
        if invalid:
            pairs = ", ".join([f"{n}='{v}'" for n, v in invalid])
            logger.warning("Predict - invalid numeric input: %s", pairs)
            return render_template('index.html', result=f" Invalid input: {pairs}")

        # PREPARE INPUT ARRAY
        input_array = np.array(values).reshape(1, -1)
        logger.debug("Predict - raw input: %s", input_array.tolist())

        # SCALING
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
                return render_template('index.html', result=" Scaler error — check compatibility.")
        else:
            logger.warning("Predict - scaler is None, using raw inputs")
            scaled = input_array

        # MODEL CHECK
        if model is None:
            logger.error("Predict - model not loaded")
            return render_template('index.html', result=" Model not loaded on server.")

        # PREDICTION
        try:
            pred = model.predict(scaled)

            if isinstance(pred, (list, tuple, np.ndarray)):
                pred_val = int(pred[0])
            else:
                pred_val = int(pred)

        except Exception as e:
            logger.exception("Model predict error: %s", e)
            return render_template('index.html', result=" Model prediction error.")

        # FINAL OUTPUT
        predicted_label = label_map.get(pred_val, "Unknown Crop")
        logger.info("Prediction -> %s (%s)", pred_val, predicted_label)

        return render_template('index.html', result=f"🌾 Recommended Crop: {predicted_label}")

    except Exception as e:
        logger.exception("Unhandled /predict error: %s", e)
        return render_template('index.html', result=" Error processing input. Check logs.")

# PUSH DATA TO THINGSPEAK (new)
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
    'temperature': 'field1',
    'soil_moisture': 'field2',
    'N': 'field3', 'n': 'field3',
    'P': 'field4', 'p': 'field4',
    'K': 'field5', 'k': 'field5',
    'ph': 'field6'
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

# REPORT (PDF/TXT)
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

#CHATBOT
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get('message', '').strip()
        user_input_lower = user_input.lower()
    except Exception:
        user_input = ''
        user_input_lower = ''

    if not user_input:
        return jsonify({"reply": "👋 Ask me anything about crops, soil, or weather!"})

    # 🔥 1. WEATHER
    if "weather" in user_input_lower:
        return jsonify({"reply": get_weather("Indore")})

    # 🔥 2. BASIC FARMING QUICK RESPONSES
    if "ph" in user_input_lower:
        return jsonify({"reply": " Ideal soil pH is 6–7.5."})

    if "nitrogen" in user_input_lower:
        return jsonify({"reply": " Nitrogen helps leaf growth. Use urea or compost."})

    #  3. WEB SEARCH (MAIN UPGRADE)
    web_result = search_web(user_input)

    return jsonify({"reply": web_result})
    # WEB SEARCH
# DEBUG INFO
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

# LOGOUT
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

#RUN APP
if __name__ == '__main__':
    # For production, run via gunicorn/uwsgi and set debug=False
    app.run(debug=True)
