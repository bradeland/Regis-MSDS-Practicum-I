"""
Original Developer: Brad Eland
Class: MSDS Practicum I Regis University
Class Dates: 2024-01-11 to 2024-03-18
Modified: 2024-03-09
"""

# Libraries needed to complete work
# https://ioflood.com/blog/pip-install-specific-version/#:~:text=Advanced%20Pip%20Commands%20and%20Options&text=Meanwhile%2C%20pip%20uninstall%20package%20will,package%20to%20its%20latest%20version.
# todo: Make sure you have the requirements.txt in your directory.  In terminal type this below, pip install -r requirements.txt
# pip install -r requirements.txt
# todo: may need to upgrade pip installer.  If so, do this in your terminal
# python.exe -m pip install --upgrade pip
# todo: this one needs to be imported through file-->settings-->project-->python interpreter and then searched for by selecting the + and looking for it
import yfinance as yf
from pandas_datareader import data as wb
from finta import TA
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import seaborn as sns

# trying to make as dynamic as possible, plug any ticker symbol (up to 3) in.
## GLOBAL VARIABLES
# sets # of days for lookback
v_days = 1200
v_start_stock = f'{v_days}d'
# default on interval according to https://python-yahoofinance.readthedocs.io/en/latest/api.html is 1 day.  Can set to hourly, 4 hour, monthly etc. . .
v_set_interval = '1d'
v_start_stock_date = dt.date.today() - dt.timedelta(days=v_days)
# todo: can modify to whatever ticker symbol(s) you want.  v_ticker_sym1 is the one this model is trying to predict!
v_ticker_sym1 = 'LUV'
v_ticker_sym2 = 'DAL'
v_ticker_sym3 = 'CL=F'
v_start = dt.date.today() - dt.timedelta(days=v_days)


def main():
    v_stock_predicting = d_get_stock_and_oil_data()
    d_neural_network(v_stock_predicting)
    v_closing_data_for_3, v_value = d_merge_data(v_start)
    d_intermarket_eda(v_closing_data_for_3)
    d_neural_network_3_stocks(v_closing_data_for_3, v_value)


def d_get_stock_and_oil_data():
    # pulling through multiple in case my merge doesn't work later on
    # Stock Symbol for what I'm trying to predict is LUV (Southwest Airlines)
    v_t1 = yf.Ticker(f'{v_ticker_sym1}')
    v_t1_data = v_t1.history(period=f'{v_start_stock}', interval=v_set_interval)
    v_t1_data_for_tech = v_t1_data
    # making lower in case was uppper (which some were uppper, some lower).  Makes easier regardless of the input and consistency if changing tickers.
    v_t1_data_for_tech.columns = v_t1_data_for_tech.columns.str.lower()
    v_t1_data_final = v_t1_data.add_prefix(f'{v_ticker_sym1}')
    # doing only 0:5 columns from DF to match up the oil later on.
    v_t1_data_final_cleaned = v_t1_data_final.iloc[:, 0:5]
    # Did min/max values so I could see actual price highs/lows
    v_max_value = v_t1_data_final_cleaned.max()
    v_min_value = v_t1_data_final_cleaned.min()
    print(v_max_value)
    print(v_min_value)
    # todo: write out something for RSI
    v_t1_data_final_cleaned_for_RSI = v_t1_data_final.iloc[:, 0:5]
    v_start = v_t1_data.index.T[0]
    v_t2 = yf.Ticker(f'{v_ticker_sym2}')
    v_t2_data = v_t2.history(period=v_start_stock, interval=v_set_interval)
    v_t2_data_final = v_t2_data.add_prefix(f'{v_ticker_sym2}')
    v_t2_data_final_cleaned = v_t2_data_final.iloc[:, 0:5]
    v_max_value_1 = v_t2_data_final_cleaned.max()
    v_min_value_1 = v_t2_data_final_cleaned.min()
    print(v_max_value_1)
    print(v_min_value_1)
    # did oil last so I could get the start date from the period above or start date from main stock to match when joining data
    yf.pdr_override()
    # default end date is "today."  Can add end= if needed
    v_t3_df = wb.DataReader(f'{v_ticker_sym3}',  start=v_start, interval=v_set_interval)
    print(v_t3_df.dtypes)
    if '=' in str(v_ticker_sym3):
        v_tick_clean = v_ticker_sym3.replace('=', '')
        v_t3_data_final = v_t3_df.add_prefix(f'{v_tick_clean}')
        v_t3_data_final_cleaned = v_t3_data_final.iloc[:, 0:5]
        v_max_value_2 = v_t3_data_final_cleaned.max()
        v_min_value_2 = v_t3_data_final_cleaned.min()
        print(v_max_value_2)
        print(v_min_value_2)
    else:
        v_t3_data_final = v_t3_df.add_prefix(f'{v_ticker_sym3}')
        v_t3_data_final_cleaned = v_t3_data_final.iloc[:, 0:5]
        v_max_value_1 = v_t2_data_final_cleaned.max()
        v_min_value_1 = v_t2_data_final_cleaned.min()
        print(v_max_value_1)
        print(v_min_value_1)
    d_eda(v_t1_data, v_t2_data, v_t3_df)
    return v_t1_data_final_cleaned


def d_intermarket_eda(v_prediction_of_3):
    # EDA/Heatmap of the 3 (Found this from here: https://www.marketcalls.in/python/build-a-correlation-matrix-using-python-pandas-and-seaborn.html)
    v_df_correlation = pd.DataFrame(v_prediction_of_3)
    v_corr_matrix = v_df_correlation.corr()
    # create a heatmap of the correlation matrix using Seaborn
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(v_corr_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax)

    # set the axis labels and title
    ax.set_xlabel('Stock Tickers')
    ax.set_ylabel('Stock Tickers')
    ax.set_title('10x10 Correlation Matrix of Stock Prices (1200-day period)')

    # display the plot
    plt.show()
    """
    This was to show correlation between the stocks.  
    """


def d_eda(data1, data2, data3):
    print(data1.tail(10))
    fig, ax = plt.subplots(2, 1)
    data1[["close", "volume"]].plot(subplots=True, layout=(2, 1), ax=ax)
    fig.suptitle(f'Stock Predicting {v_ticker_sym1}')
    plt.show()
    fig, ax = plt.subplots(2, 1)
    data2[["Close", "Volume"]].plot(subplots=True, layout=(2, 1), ax=ax)
    fig.suptitle(f'Help Predicting Stock {v_ticker_sym2}')
    plt.show()
    fig, ax = plt.subplots(2, 1)
    data3[["Close", "Volume"]].plot(subplots=True, layout=(2, 1), ax=ax)
    fig.suptitle(f'Futures Help Predicting {v_ticker_sym3}')
    plt.show()


def d_modeling_machine(v_stock_predict):
    v_stock_predict_final = v_stock_predict.reset_index()
    v_stock_predict_final['Date'] = pd.to_datetime(v_stock_predict_final['Date']).dt.date
    v_stock_predict_final['Date'] = (v_stock_predict_final['Date'] - v_stock_predict_final['Date'].min()) / np.timedelta64(1, 'D')
    # Step 2 & 3: Feature and target variable selection
    X = v_stock_predict_final[['Date']]  # Features
    y = v_stock_predict_final[f'{v_ticker_sym1}close']  # Target variable

    # Step 4: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Step 5: Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    score = model.score(X_test, y_test)
    print(f'Model accuracy: {score:.2f}')

    # Step 7: Predict
    # Predict the closing prices for the dates in X_test
    predictions = model.predict(X_test)

    # Plotting the results
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, predictions, color='blue', linewidth=3, label='Linear regression')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()


def d_neural_network(v_stock_predict):
    # Assuming df['Close'] is the closing price
    prices = v_stock_predict[f'{v_ticker_sym1}close'].values
    # Parameters
    n_past = 60  # Number of past days to consider for predicting the next day's price

    # Create sequences (X) and targets (y)
    X, y = [], []
    for i in range(n_past, len(prices)):
        X.append(prices[i - n_past:i])  # Sequence of past prices
        y.append(prices[i])  # Next day's price
    X, y = np.array(X), np.array(y)
    # todo: got x/y up to here :) FINAFREAKINLY! :d

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Assuming X_train and X_test are numpy arrays
    # Reshape for LSTM [Data, Time, Features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout

    # Define the LSTM model
    model = Sequential()

    # Adding the first LSTM layer and some Dropout regularization
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularization
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    # Making predictions
    predicted_stock_price = model.predict(X_test)

    # Plotting the actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Actual Stock Price')
    plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, predicted_stock_price))
    print("RMSE:", rmse)


def d_merge_data(start_date):
    # Todo: currently using close.  This can be adjusted to 'Adj Close_', 'High_', 'Low_', 'Open_', 'Volume_'
    v_value_using_to_predict = 'Close_'
    # https://www.youtube.com/watch?v=Ag_1Ysqw2J4 merge easily!
    stock = [f'{v_ticker_sym1}', f'{v_ticker_sym2}', f'{v_ticker_sym3}']
    df_merged = yf.download(stock, start=start_date)
    print(df_merged.tail(10))
    # df_merged_with_date = df_merged.reset_index()
    for d in df_merged.columns:
        v_tuple_name = d
        v_target_name = '_'.join(v_tuple_name)
        # Replace the tuple-named column with the new string name
        df_merged.columns = [v_target_name if col == d else col for col in df_merged.columns]
    df_merged_final = df_merged.dropna()
    # Reset the index to bring in the date to the dataset
    df_merged_final_with_date = df_merged_final.reset_index()

    # Plots the merged stocks
    plt.figure(figsize=(10, 6))
    plt.plot(df_merged_final_with_date[f'{v_value_using_to_predict}{v_ticker_sym1}'], label=f"Stock Predicting")
    plt.plot(df_merged_final_with_date[f'{v_value_using_to_predict}{v_ticker_sym2}'], label=f"Same Industry as Prediction Stock")
    plt.plot(df_merged_final_with_date[f'{v_value_using_to_predict}{v_ticker_sym3}'], label=f"Futures Stock")

    # Add titles and labels
    plt.title('Stocks over time')
    plt.xlabel('# of Days')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show

    df_merged_closing = df_merged_final_with_date[['Date', f'{v_value_using_to_predict}{v_ticker_sym1}', f'{v_value_using_to_predict}{v_ticker_sym2}', f'{v_value_using_to_predict}{v_ticker_sym3}']]

    # adjusted close definition: https://help.yahoo.com/kb/SLN28256.html#:~:text=Adjusted%20close%20is%20the%20closing,applicable%20splits%20and%20dividend%20distributions.
    # explanation on why I'm not going to use this.  Looking for shorter term trades. https://www.reddit.com/r/investing/comments/1gv2hr/yahoo_finance_closing_price_vs_adjusted_price/
    return df_merged_closing, v_value_using_to_predict


def d_neural_network_3_stocks(v_predictor_of_3, v_value_passed):
    # todo: make sure you have your 2nd/3rd stocks in x
    X = v_predictor_of_3[[f'{v_value_passed}{v_ticker_sym2}', f'{v_value_passed}{v_ticker_sym3}']].values
    y = v_predictor_of_3[f'{v_value_passed}{v_ticker_sym1}']
    y_final = y.to_numpy()
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = y_final.reshape(-1, 1)
    y_scaled = scaler.fit_transform(y)
    print(y_scaled)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Building the model (added a couple of dropouts to make sure the model isn't over-fit [meaning the NN can predict the price too easily and not learn new data])
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    # .15 with .2 dropout
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # Predict using the model (example)
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    print(predicted_prices)

    # Convert predictions and actual values to their original scale if they were normalized
    # This step is necessary only if your target variable was normalized before training
    actual_prices = scaler.inverse_transform(y_test)
    predicted_prices = scaler.inverse_transform(model.predict(X_test))

    # Plotting the data
    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    plt.scatter(range(len(actual_prices)), actual_prices, color='blue', label='Actual', alpha=0.5)
    plt.scatter(range(len(predicted_prices)), predicted_prices, color='red', label='Predicted', alpha=0.5)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Plotting with lines
    plt.figure(figsize=(12, 6))  # Set the figure size for better readability
    plt.plot(actual_prices, label='Actual', color='blue', marker='o')
    plt.plot(predicted_prices, label='Predicted', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(y_test, predictions))
    print("RMSE:", rmse)


    # Todo: This helps w/getting columns I need from DF. https://www.statology.org/pandas-select-column-by-index/#:~:text=If%20you'd%20like%20to,loc%20function.

    # https://towardsdatascience.com/algorithmic-trading-with-macd-and-python-fef3d013e9f3#:~:text=The%20MACD%20is%20calculated%20by,it%20would%20be%20an%20uptrend.

    # This script downloads the data, and then it calculates the macd values such as the signal and the histogram that defines the trend.


if __name__ == "__main__":
    main()