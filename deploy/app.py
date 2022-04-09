import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load(open('models/model.h5', "rb"))
scaler = joblib.load(open('models/scaler.h5', "rb"))
seasons = {
    'winter': [0, 0, 1],
    'spring': [1, 0, 0],
    'summer': [0, 1, 0],
    'fall': [0, 0, 0]
}

weathers = {
    'clear': [0, 0, 0],
    'mist': [1, 0, 0],
    'rainy': [0, 1, 0],
    'snowy': [0, 0, 1]
}


weekdays = {
    'saturday': [0, 1, 0, 0, 0, 0],
    'sunday': [0, 0, 1, 0, 0, 0],
    'monday': [1, 0, 0, 0, 0, 0],
    'tuesday': [0, 0, 0, 0, 1, 0],
    'wednesday': [0, 0, 0, 0, 0, 1],
    'thursday': [0, 0, 0, 1, 0, 0],
    'friday': [0, 0, 0, 0, 0, 0]
}



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/predict', methods=['GET'])
def predict():
    is_holiday = int(request.args.get('is_holiday', 0))
    temp = float(request.args.get('temp'))
    humidity = float(request.args.get('humidity'))
    rented = int(request.args.get('rented_bikes'))
    month = int(request.args.get('month'))
    hour = int(request.args.get('hour'))
    season = seasons[request.args.get('season')]
    weather = weathers[request.args.get('weather')]
    weekday = weekdays[request.args.get('day')]

    pre = [is_holiday, temp, humidity,rented, hour, month]
    pre += season + weather + weekday

    pre = scaler.transform([pre])
    profit = model.predict(pre)[0]
    profit = round(profit,2)
    return render_template('index.html', prediction=profit)



if __name__ == "__main__":
    app.run(debug=True)