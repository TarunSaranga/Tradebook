import datetime as dt
import pandas as pd
import numpy as np
from utilgog import get_data
import pytz
import math

import time
import gc
import sklearn

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import average_precision_score, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

look_after_days = 14
rise_percent_threshold = 1
cash_max = 1000000
max_entries = 10

x_test_svc = None
x_test_svr = None

plot_entries = True

def trade(x_test, y_pred_test, model = 'svc'):
    cash = cash_max
    entries = 0
    shares_holding = 0

    bought = 'bought'
    shares = 'shares'
    open = 'open'
    close = 'close'

    portfolio = 'portfolio'
    hold_portfolio = 'hold_portfolio'

    x_test[bought] = 0
    x_test[portfolio] = cash_max
    x_test[hold_portfolio] = cash_max

    open_price = x_test.iloc[0][open]
    invest_shares_cnt = int(cash_max/open_price)
    invest_cash_left = cash_max - invest_shares_cnt*open_price

    for i in np.arange(x_test.shape[0]):
        # sell any previous entry
        if i >= look_after_days and x_test.iloc[i - look_after_days][bought] == 1:
            shares_to_sell = x_test.iloc[i - look_after_days][shares]
            cash += shares_to_sell*x_test.iloc[i][open]
            entries -= 1
            shares_holding -= shares_to_sell

        # reenter using buy, if possible
        thres = 1 if model == 'svc' else rise_percent_threshold
        if model == 'svc':
            assert thres == 1
        elif model == 'svr':
            assert thres == rise_percent_threshold
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

        close_price = x_test.iloc[i][close]
        x_test.loc[x_test.index[i], portfolio] = cash + shares_holding*close_price
        x_test.loc[x_test.index[i], hold_portfolio] = invest_cash_left + invest_shares_cnt*close_price

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

    if model == 'svc':
        x_test_svc = x_test.copy()
    elif model == 'svr':
        x_test_svr = x_test.copy()

    # Plot portfolio value
    df = pd.concat([x_test[portfolio], x_test[hold_portfolio]], keys=['Portfolio - {}'.format(model.upper()), 'Portfolio - Buy and Hold'], axis=1)
    ax = df.plot(title='{} - Portfolio Value {} (Test Set)'.format(model.upper(), "with good/bad trades" if plot_entries is True else ""), fontsize=12, color=['#0000FF', 'orange'])
    if plot_entries is True:
        for i in np.arange(x_test.shape[0]):
            if x_test.iloc[i][bought] == 1:
                loss = False
                if i+look_after_days < x_test.shape[0] and x_test.iloc[i][open] > x_test.iloc[i+look_after_days][open]:
                    loss = True
                ax.axvline(x_test.index[i], color="red" if loss is True else "green", linestyle="--")


    plt.savefig('{}_{}.png'.format(model, "trades" if plot_entries is True else ""), dpi=200)
    plt.close()


def svc_results(data):
    # Separate out the x_data and y_data.

    y_data = data.loc[:, 'rose_above_threshold']
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 100, shuffle = False)

    x_data_date = data.iloc[:, data.columns != 'rose_above_threshold']
    x_train_date, x_test_date, y_train, y_test = train_test_split(x_data_date, y_data, test_size = 0.2, random_state = 100, shuffle = False)

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

    x_data_date = data.iloc[:, data.columns != 'rose_above_threshold']
    x_data_date = x_data_date.iloc[:, x_data_date.columns != 'up']
    x_train_date, x_test_date, y_train, y_test = train_test_split(x_data_date, y_data, test_size = 0.2, random_state = 100, shuffle = False)

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

    # print('y_test = '  + str(y_test[:5]))
    # print('y_pred_test = '  + str(y_pred_test[:5]))

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




#def process(symbol="GOG_mod", sd=dt.datetime(2015, 1, 1), ed=dt.datetime(2017, 2, 1)):
def process(symbol="GOG_mod", sd=dt.datetime(2010, 8, 12), ed=dt.datetime(2018, 8, 9), plot_trades = True):
    global plot_entries
    plot_entries = plot_trades
    syms = [symbol]
    df = get_data(syms, pd.date_range(sd, ed))

    print('df.shape = ' + str(df.shape))

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

    print('#########SVC results for GOG_mod.csv#############')
    svc_results(df)
    print('#############################################\n')

    print('#########SVR results for GOG_mod.csv#############')
    svr_results(df)
    print('#############################################')

if __name__ == "__main__":
    process(symbol="GOG_mod", sd=dt.datetime(2010, 8, 12), ed=dt.datetime(2018, 8, 9), plot_trades = False)
    process(symbol="GOG_mod", sd=dt.datetime(2015, 1, 1), ed=dt.datetime(2017, 2, 1))