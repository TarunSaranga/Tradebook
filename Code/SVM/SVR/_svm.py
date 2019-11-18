import datetime as dt
import pandas as pd
import numpy as np
from util import get_data
import pytz
import math

import time
import gc
import sklearn

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, mean_squared_error

time_zone_ind_str = 'Asia/Calcutta'

look_after_days = 14
rise_percent_threshold = 0.2
cash_max = 100000
max_entries = 10

def custom_round(arr):
    c = np.copy(arr)
    c[c >= 0.5] = 1
    c[c < 0.5] = 0
    return c


def trade(x_test, y_pred_test, model = 'svc'):
    cash = cash_max
    entries = 0
    shares_holding = 0

    bought = 'bought'
    bought_idx = 8
    shares = 'shares'
    shares_idx = 9
    open_idx = 0
    open = 'open'
    close = 'close'

    x_test[bought] = 0

    for i in np.arange(x_test.shape[0]):
        # sell any previous entry
        if i >= look_after_days and x_test.iloc[i - look_after_days][bought] == 1:
            shares_to_sell = x_test.iloc[i - look_after_days][shares]
            cash += shares_to_sell*x_test.iloc[i][open_idx]
            entries -= 1
            shares_holding -= shares_to_sell

        # reenter using buy, if possible
        thres = 1 if model == 'svc' else rise_percent_threshold
        if y_pred_test[i] >= thres and entries < max_entries:
            cash_to_use = min(cash, cash_max/max_entries)
            open_price = x_test.iloc[i][open]
            shares_to_buy = int(cash_to_use/open_price)
            if shares_to_buy > 0:
                entries += 1
                cash -= shares_to_buy*open_price
                x_test.loc[x_test.index[i], bought] = 1
                assert (x_test.iloc[i][bought] == 1)
                x_test.loc[x_test.index[i], shares] = shares_to_buy
                assert (x_test.iloc[i][shares] == shares_to_buy)
                shares_holding += shares_to_buy

    print('shares_holding at end of trading = ' + str(shares_holding))
    portfolio_value = cash + shares_holding*x_test.iloc[-1][close]
    print('portfolio_value = ' + str(portfolio_value))

    # Calculate portfolio_value if you had bought and hold all this while
    open_price = x_test.iloc[0][open]
    close_price = x_test.iloc[-1][close]
    shares_holding = int(cash_max/open_price)
    cash = cash_max - shares_holding*open_price
    portfolio_value = cash + shares_holding*close_price
    print('shares_holding during investment = ' + str(shares_holding))
    print('portfolio_value (buy and invest without trading) = ' + str(portfolio_value))


def svc_results(data):
    # Separate out the x_data and y_data.

    y_data = data.loc[:, 'rose_above_threshold']
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 100, shuffle = False)

    x_data_date = data.iloc[:, data.columns != 'rose_above_threshold']
    x_train_date, x_test_date, y_train, y_test = train_test_split(x_data_date, y_data, test_size = 0.3, random_state = 100, shuffle = False)

    x_train = x_train_date.iloc[:, 0:]
    x_test = x_test_date.iloc[:, 0:]

    #print('x_train = '  + str(x_train))
    #print('x_test = '  + str(x_test))

    x_train_scaled = StandardScaler().fit_transform(x_train)
    x_test_scaled = StandardScaler().fit_transform(x_test)

    svc = SVC(gamma='auto')
    #svc.fit(x_train, y_train)
    svc.fit(x_train_scaled, y_train)
    y_pred_train = svc.predict(x_train_scaled)
    train_score = accuracy_score(y_train, y_pred_train)

    y_pred_test = svc.predict(x_test_scaled)
    test_score = accuracy_score(y_test, y_pred_test)

    #print('y_test = '  + str(y_test))
    #print('y_pred_test = '  + str(y_pred_test))

    print('train_score = '  + str(train_score))
    print('test_score = '  + str(test_score))
    print('average_precision_train_score = '  + str(average_precision_score(y_train, y_pred_train)))
    print('average_precision_test_score = '  + str(average_precision_score(y_test, y_pred_test)))

    # SVM done

    # Simulate trading and calculate profits with starting cash of 100000
    trade(x_test, y_pred_test, model='svc')

def svr_results(data):
    # Separate out the x_data and y_data.

    y_data = data.loc[:, 'up']
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 100, shuffle = False)

    x_data_date = data.iloc[:, data.columns != 'rose_above_threshold']
    x_data_date = x_data_date.iloc[:, x_data_date.columns != 'up']
    x_train_date, x_test_date, y_train, y_test = train_test_split(x_data_date, y_data, test_size = 0.3, random_state = 100, shuffle = False)

    x_train = x_train_date.iloc[:, 0:]
    x_test = x_test_date.iloc[:, 0:]

    #print('x_train = '  + str(x_train))
    #print('x_test = '  + str(x_test))

    x_train_scaled = StandardScaler().fit_transform(x_train)
    x_test_scaled = StandardScaler().fit_transform(x_test)

    svr = SVR(gamma='auto')
    svr.fit(x_train_scaled, y_train)
    y_pred_train = svr.predict(x_train_scaled)
    #train_score = svr.score(x_train_scaled, y_pred_train)

    y_pred_test = svr.predict(x_test_scaled)
    #test_score = svr.score(x_test_scaled, y_pred_test)

    print('y_test = '  + str(y_test[:5]))
    print('y_pred_test = '  + str(y_pred_test[:5]))

    # print('train_score = '  + str(train_score))
    # print('test_score = '  + str(test_score))

    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = math.sqrt(mse_train)
    print("Train Root Mean Squared Error:", rmse_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = math.sqrt(mse_test)
    print("Test Root Mean Squared Error:", rmse_test)

    # SVR done

    # Simulate trading and calculate profits with starting cash of 100000
    trade(x_test, y_pred_test, model='svr')





def process(symbol="JustDial", sd=dt.datetime(2018, 1, 1).replace(tzinfo=pytz.timezone(time_zone_ind_str)), ed=dt.datetime(2018, 1, 30).replace(tzinfo=pytz.timezone(time_zone_ind_str))):
    syms = [symbol]
    df = get_data(syms, pd.date_range(sd, ed, freq='min', tz=time_zone_ind_str))  # automatically adds SPY

    open = df['open']
    a = np.zeros(df.shape[0])
    #a[look_after_days:] =((open[look_after_days:] - open[:-look_after_days].values) / open[:-look_after_days].values)*100
    a[:-look_after_days] =((open[look_after_days:] - open[:-look_after_days].values) / open[:-look_after_days].values)*100

    # price rise in %age compared to the price 'look_after_days' in future
    df['up'] = a

    df = df[:-look_after_days]
    rose_above_threshold = (df['up'] >= rise_percent_threshold)
    rose_above_threshold[rose_above_threshold == True] = 1
    df['rose_above_threshold'] = rose_above_threshold

    #print(len(df))
    #print(df)

    #svc_results(df)
    svr_results(df)

if __name__ == "__main__":
    process()