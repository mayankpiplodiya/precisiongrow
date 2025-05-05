from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
import traceback

app = Flask(__name__)
app.secret_key = 'secretkey'

# Load the trained model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))  # load the scaler used during training
except Exception as e:
    print("Error loading model or scaler:", e)
    model = None
    scaler = None

users = {'admin': 'admin'}

# Crop label mapping (based on model output class index)
label_map = {
    0: "rice", 1: "maize", 2: "chickpea", 3: "kidneybeans", 4: "pigeonpeas",
    5: "mothbeans", 6: "mungbean", 7: "blackgram", 8: "lentil", 9: "pomegranate",
    10: "banana", 11: "mango", 12: "grapes", 13: "watermelon", 14: "muskmelon",
    15: "apple", 16: "orange", 17: "papaya", 18: "coconut", 19: "cotton",
    20: "jute", 21: "coffee"
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid Credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return render_template('signup.html', error='Username already exists')
        users[username] = password
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

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

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and convert input values
            features = [float(x) for x in request.form.values()]
            print("Received input values:", features)

            # Preprocess input using the same scaler as model training
            input_array = np.array(features).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            print("Scaled input:", scaled_input)

            # Predict using the model
            prediction = model.predict(scaled_input)
            print("Raw prediction:", prediction)

            predicted_label = label_map.get(prediction[0], "Unknown Crop")
            return render_template('index.html', result=f"Recommended Crop: {predicted_label}")
        except Exception as e:
            traceback.print_exc()
            return render_template('index.html', result="Invalid input or server error.")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

