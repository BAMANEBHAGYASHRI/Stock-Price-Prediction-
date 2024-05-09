import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()
from keras.models import load_model
import streamlit as st 

startdate = datetime(2010,1,1)
enddate = datetime(2022,12,31)

st.title('Stock Price Prediction')

y_symbols= st.text_input('Enter stock ticker', 'AAPL')
data = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)

st.subheader('Data from 2010 - 2019')
st.write(data.describe())

st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(data.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past_100_days = data_testing.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])
  
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Prediction vs Original ')
from pandas.io.formats.style import plt
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'pridected price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)