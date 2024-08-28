from flask import Flask, request, jsonify,render_template
import joblib
import pandas as pd
import os
app = Flask(__name__)
file_path = 'model.pkl'
print(os.getcwd())
print(os.path.exists(file_path))
# Load the trained model
model = joblib.load('model.pkl')
@app.route("/")
def root():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run()


