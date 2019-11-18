"""MLT: Utility code.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import os
import pandas as pd

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", '../Data/')
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    #print(df.index)
    date_clm = 'DateTime'

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col=date_clm,
                              parse_dates=True, na_values=['nan'])
        #df_temp.index = df_temp.index.tz_convert('Asia/Calcutta')
        #df_temp = df_temp.rename(columns={colname: symbol})
        #df_temp[date_clm] = df_temp.index
        #df_temp[date_clm] = df_temp[date_clm].dt.tz_localize(None)
        #print(df_temp)
        #print(df.merge(df_temp, right_index=True, left_index=True))
        df = df.merge(df_temp, right_index=True, left_index=True)
        #df = df.join(df_temp)

    #df.index = df.index.tz_convert('Asia/Calcutta')
    #print(df.index.tz_convert('Asia/Calcutta'))
    #print(df.index)
    return df

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def get_orders_data_file(basefilename):
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR",'orders/'),basefilename))

def get_learner_data_file(basefilename):
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR",'Data/'),basefilename),'r')

def get_robot_world_file(basefilename):
    return open(os.path.join(os.environ.get("ROBOT_WORLDS_DIR",'testworlds/'),basefilename))
