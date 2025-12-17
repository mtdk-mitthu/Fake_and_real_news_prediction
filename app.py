from flask import Flask, request, jsonify
import joblib
import traceback
import sys
import os

app = Flask(__name__)

# Load trained model if available
model_filename = 'model.pkl'
if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    print('Model loaded successfully')
else:
    model = None
    print('Model not found')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            prediction = model.predict(json_)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return jsonify({'error': 'No model loaded'})

# Run Flask app locally
if __name__ == "__main__":
    port = 12345
    app.run(port=port, debug=True)
