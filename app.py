from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import joblib


app = Flask(__name__)

# Model ve pipeline'ı yükleme
model = joblib.load('model.pkl')
full_pipeline = joblib.load("pipeline.pkl")



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
    query = 'SELECT * FROM emlak_verileri WHERE city=?'
    params = [city]

    if district:
        query += ' AND district=?'
        params.append(district)
    if neighborhood:
        query += ' AND neighborhood=?'
        params.append(neighborhood)

    emlak = conn.execute(query, params).fetchall()
    conn.close()

    emlak_list = [dict(ix) for ix in emlak]
    return jsonify(emlak_list)

##@app.route('/emlak_district', methods=['GET'])
##def get_emlak_district():
##    city = request.args.get('city')
##    district = request.args.get('district')
##    conn = get_db_connection()
##    query = 'SELECT * FROM emlak_verileri WHERE city=? AND district=? '
##    emlak = conn.execute(query, (city, district)).fetchall()
##    conn.close()
##    emlak_list = [dict(ix) for ix in emlak]
##    return jsonify(emlak_list)
##
##
##
##@app.route('/emlak_detail', methods=['GET'])
##def get_emlak_detail():
##    city = request.args.get('city')
##    district = request.args.get('district')
##    neighborhood = request.args.get('neighborhood')
##
##    conn = get_db_connection()
##    query = 'SELECT * FROM emlak_verileri WHERE city=? AND district=? AND neighborhood=? '
##    emlak = conn.execute(query, (city, district,neighborhood)).fetchall()
##    conn.close()
##    emlak_list_detail = [dict(ix) for ix in emlak]
##    return jsonify(emlak_list_detail)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    df = pd.DataFrame([data])
    print("DataFrame:\n", df)
    print(df.info())
    prepared_data = full_pipeline.transform(df)
    print("Prepared data:\n", prepared_data)
    print(prepared_data.shape)
    prediction = model.predict(prepared_data)
    print("Prediction:", prediction)
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
