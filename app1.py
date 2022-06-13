import warnings
warnings.filterwarnings('ignore')  # Hide warnings
#import datetime as dt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#from PIL import Image
#import os
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import streamlit as st

#title
st.title('Stock Market Analyzer')
'---------------------------------------------------------'
#st.write("Developed by Ashutosh & Koshal Garg")

com = st.text_input("Enter the Stock Code of company","AAPL")
'You Enterted the company code: ', com
st_date= st.text_input("Enter Starting date as YYYY-MM-DD", "2021-01-10")
'You Enterted the starting date: ', st_date
end_date= st.text_input("Enter Ending date as YYYY-MM-DD", "2022-01-20")
'You Enterted the ending date: ', end_date

df = web.DataReader(com, 'yahoo', st_date, end_date)  # Collects data
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)

#title
st.title('Stock Market Data')
'The Complete Stock Data as extracted from Yahoo Finance: '
df
'1. The Stock Open Values over time: '
st.line_chart(df["Open"])
'2. The Stock Close Values over time: '
st.line_chart(df["Close"])

mov_avg= st.text_input("Enter number of days Moving Average:", "50")
'You Enterted the Moving Average: ', mov_avg
df["mov_avg_close"] = df['Close'].rolling(window=int(mov_avg),min_periods=0).mean()
'1. Plot of Stock Closing Value for '+ mov_avg+ " Days of Moving Average"
'   Actual Closing Value also Present'
st.line_chart(df[["mov_avg_close","Close"]])
df["mov_avg_open"] = df['Open'].rolling(window=int(mov_avg),min_periods=0).mean()
'2. Plot of Stock Open Value for '+ mov_avg+ " Days of Moving Average"
'   Actual Opening Value also Present'
st.line_chart(df[["mov_avg_open","Open"]])



ohlc_day= st.text_input("Enter number of days for Resampling for OHLC CandleStick Chart", "50")
# Resample to get open-high-low-close (OHLC) on every n days of data
df_ohlc = df.Close.resample(ohlc_day+'D').ohlc() 
df_volume = df.Volume.resample(ohlc_day+'D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc.Date = df_ohlc.Date.map(mdates.date2num)
# Create and visualize candlestick charts
plt.figure(figsize=(8,6))
'OHLC Candle Stick Graph for '+ ohlc_day+ " Days"
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
plt.xlabel('Time')
plt.ylabel('Stock Candle Sticks')
#import warnings
#warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 7))
plt.subplots_adjust(top=1.25, bottom=1.2)

plt.subplot(2,2, 1)
df['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel('Date')
plt.title(f"Sales Volume for {com}")

st.pyplot(plt.tight_layout())


st.header('Daily returns')
# We'll use pct_change to find the percent change for each day
df["Daily Return"] = df["Adj Close"].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_figheight(15)
fig.set_figwidth(20)

df["Daily Return"].plot(ax=axes, legend=True, linestyle='--', marker='o')
axes.set_title(com)

fig.tight_layout()
st.pyplot(fig)

# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len



st.header('Prediction')
# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#scaled_data



# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len)+100, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape



from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)



# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
#rmse



# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(plt.show())