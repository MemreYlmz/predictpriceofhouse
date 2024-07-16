from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import pickle

app = Flask(__name__)

# Model ve pipeline'ı yükleme
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pipeline.pkl', 'rb') as f:
    full_pipeline = pickle.load(f)

def get_db_connection():
    conn = sqlite3.connect('hepsiemlak.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/emlak', methods=['GET'])
def get_emlak():
    city = request.args.get('city')
    district = request.args.get('district')
    neighborhood = request.args.get('neighborhood')

    conn = get_db_connection()
    query = 'SELECT * FROM emlak_verileri WHERE city=? AND district=? AND neighborhood=?'
    emlak = conn.execute(query, (city, district, neighborhood)).fetchall()
    conn.close()

    emlak_list = [dict(ix) for ix in emlak]
    return jsonify(emlak_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prepared_data = full_pipeline.transform(df)
    prediction = model.named_steps["model"].predict(prepared_data) #named_steps önemli
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
