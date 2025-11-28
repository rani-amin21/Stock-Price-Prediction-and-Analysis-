import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

# Set plot style
plt.style.use("fivethirtyeight")

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained LSTM model
model = load_model('stock_dl_model.h5')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'RELIANCE.NS'  # default fallback

        # Download stock data
        start = dt.datetime(2014, 1, 1)
        end = dt.datetime.now()
        df = yf.download(stock, start=start, end=end)

        # ✅ Check if data is empty or "Close" column missing
        if df.empty or 'Close' not in df.columns:
            return render_template("index.html", prediction="❌ Invalid stock symbol or no data found.")

        # Descriptive statistics
        data_desc = df.describe()

        # Moving Averages
        ema20 = df['Close'].ewm(span=20).mean()
        ema50 = df['Close'].ewm(span=50).mean()
        ema100 = df['Close'].ewm(span=100).mean()
        ema200 = df['Close'].ewm(span=200).mean()

        # Prepare data
        data = df[['Close']]
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * 0.7))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Split data
        train_data = scaled_data[0:training_data_len, :]
        x_train, y_train = [], []
        for i in range(100, len(train_data)):
            x_train.append(train_data[i - 100:i, 0])
            y_train.append(train_data[i, 0])
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Test data
        test_data = scaled_data[training_data_len - 100:, :]
        x_test, y_test = [], dataset[training_data_len:, :]
        for i in range(100, len(test_data)):
            x_test.append(test_data[i - 100:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predict
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # ✅ Plot 1: EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Close'], label='Close Price', linewidth=1)
        ax1.plot(ema20, label='EMA 20', linewidth=1)
        ax1.plot(ema50, label='EMA 50', linewidth=1)
        ax1.set_title('Close Price with EMA 20 & EMA 50')
        ax1.legend()
        path1 = 'static/ema_20_50.png'
        fig1.savefig(path1)
        plt.close(fig1)

        # ✅ Plot 2: EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df['Close'], label='Close Price', linewidth=1)
        ax2.plot(ema100, label='EMA 100', linewidth=1)
        ax2.plot(ema200, label='EMA 200', linewidth=1)
        ax2.set_title('Close Price with EMA 100 & EMA 200')
        ax2.legend()
        path2 = 'static/ema_100_200.png'
        fig2.savefig(path2)
        plt.close(fig2)

        # ✅ Plot 3: Actual vs Predicted
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, label="Actual", linewidth=1)
        ax3.plot(predictions, label="Predicted", linewidth=1)
        ax3.set_title("Predicted vs Actual Stock Prices")
        ax3.legend()
        path3 = 'static/pred_vs_actual.png'
        fig3.savefig(path3)
        plt.close(fig3)

        # Save dataset to CSV
        csv_path = f'static/{stock}_data.csv'
        df.to_csv(csv_path)

        return render_template("index.html",
                               plot_path_ema_20_50=path1,
                               plot_path_ema_100_200=path2,
                               plot_path_prediction=path3,
                               data_desc=data_desc.to_html(classes='table table-striped'),
                               dataset_link=csv_path,
                               prediction=f"✅ Prediction completed for {stock}")

    return render_template("index.html")


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
