from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load pre-fitted encoders and model
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_xgb_model.pkl')

app = Flask(__name__)

def preprocess_new_data(new_data, one_hot_encoder, scaler):
    encoded = one_hot_encoder.transform(new_data[['SEX', 'EDUCATION', 'MARRIAGE']])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['SEX', 'EDUCATION', 'MARRIAGE']))
    
    scaled_df = pd.DataFrame(scaler.transform(new_data[['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                                        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                                                        'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
                                                        'PAY_AMT6']]),
                             columns=['LIMIT_BAL', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                                      'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                                      'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
    
    preprocessed_data = pd.concat([encoded_df, scaled_df], axis=1)
    
    return preprocessed_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_data = pd.DataFrame(data)
    preprocessed_data = preprocess_new_data(new_data, one_hot_encoder, scaler)
    predictions = model.predict(preprocessed_data)
    messages = ['The customer is likely to default' if pred == 1 else 'The customer will not default' for pred in predictions]
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True)
