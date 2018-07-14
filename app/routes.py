from app import app
from flask import render_template, request
from forecast import utilities, linearregressionmodel


@app.route('/')
@app.route('/index')
def index():

    return render_template('home.html')


@app.route('/stocks')
def stocks():
    utils = utilities.Utilities()
    company = request.args.get('company')
    data = utils.get_data(company)
    X_train, X_test, y_train, y_test = utils.pre_process(data)

    model = linearregressionmodel.LinearRegressionModel()
    accuracy = model.train(X_train, y_train, X_test, y_test)

    prediction = model.predict(utils.get_recent())

    formatted_data = utils.get_formatted_data()
    predicted_forecast = utils.get_predicted_forecast(prediction)

    return render_template('stocks.html',
                           company=company,
                           data=prediction,
                           accuracy=accuracy,
                           formatted_data=formatted_data,
                           predicted_forecast=predicted_forecast)