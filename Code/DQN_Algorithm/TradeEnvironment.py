import pandas as pd
import numpy as np
from tensorforce.environments import Environment
import matplotlib.pyplot as plt


class TradeEnvironment(Environment):

    def __init__(self,start_date="2018-01-01",end_date="2018-02-01",
                 starting_cash=1000000,charges=False,viz=False):

        self.sd = start_date
        self.ed = end_date
        self.starting_cash = starting_cash
        self.stock_size = 100
        df = pd.read_csv("../Data/JustDial.csv",index_col="DateTime")
        df.index = pd.to_datetime(df.index)
        self.data = df[self.sd:self.ed]
        self._max_timesteps = self.data.shape[0]
        self.position = 0
        self.profit = 0
        self.data["BuySignal"] = np.nan
        self.data["SellSignal"] = np.nan
        #self.data["HoldSignal"] = np.nan
        self.viz = viz
        ######################################################################
        #reset values
        self.time_step = 10
        self.profit = 0
        self.curr_cash = self.starting_cash
        self.prev_buy = 0
        self.curr_price = self.data["close"][self.time_step]
        self.prev1_price = self.data["close"][self.time_step-1]
        self.prev2_price = self.data["close"][self.time_step-2]
        self.prev3_price = self.data["close"][self.time_step-3]
        self.prev4_price = self.data["close"][self.time_step-4]
        self.prev5_price = self.data["close"][self.time_step-5]
        self.prev6_price = self.data["close"][self.time_step-6]
        self.prev7_price = self.data["close"][self.time_step-7]
        self.prev8_price = self.data["close"][self.time_step-8]
        self.prev9_price = self.data["close"][self.time_step-9]
        self.prev10_price = self.data["close"][self.time_step-10]
        self.close_sma = self.data["close_sma"][self.time_step]
        self.bb = self.data["bb"][self.time_step]
        self.rsi = self.data["rsi"][self.time_step]

        ######################################################################

        self._actions = dict(type="int",shape=(),num_values=2)
        self._states = dict(type="float",shape=(13))
#        self._states = dict(
#                            curr_holding  = dict(shape=(1),type='float'),
#                            curr_price = dict(shape=(1),type='float'),
#                            curr_position = dict(shape=(1),type='float'),
#                            curr_closesma = dict(shape=(1),type='float'),
#                            curr_bb = dict(shape=(1),type='float'),
#                            curr_rsi = dict(shape=(1),type='float')
#                            )


    def actions(self):
        return self._actions

    def states(self):
        return self._states

    def max_episode_timesteps(self):
        return self._max_timesteps

    def reset(self):
        self.time_step = 10
        self.curr_cash = self.starting_cash
        self.prev_buy = 0
        self.curr_price = self.data["close"][self.time_step]
        self.prev1_price = self.data["close"][self.time_step-1]
        self.prev2_price = self.data["close"][self.time_step-2]
        self.prev3_price = self.data["close"][self.time_step-3]
        self.prev4_price = self.data["close"][self.time_step-4]
        self.prev5_price = self.data["close"][self.time_step-5]
        self.prev6_price = self.data["close"][self.time_step-6]
        self.prev7_price = self.data["close"][self.time_step-7]
        self.prev8_price = self.data["close"][self.time_step-8]
        self.prev9_price = self.data["close"][self.time_step-9]
        self.prev10_price = self.data["close"][self.time_step-10]
        self.close_sma = self.data["close_sma"][self.time_step]
        self.bb = self.data["bb"][self.time_step]
        self.rsi = self.data["rsi"][self.time_step]
        self.data["BuySignal"] = np.nan
        self.data["SellSignal"] = np.nan
        #self.data["HoldSignal"] = np.nan
        next_state = self.get_next_state()

        return next_state

    def update_time_step(self):
        self.time_step += 1
        self.curr_price = self.data["close"][self.time_step]
        self.prev1_price = self.data["close"][self.time_step-1]
        self.prev2_price = self.data["close"][self.time_step-2]
        self.prev3_price = self.data["close"][self.time_step-3]
        self.prev4_price = self.data["close"][self.time_step-4]
        self.prev5_price = self.data["close"][self.time_step-5]
        self.prev6_price = self.data["close"][self.time_step-6]
        self.prev7_price = self.data["close"][self.time_step-7]
        self.prev8_price = self.data["close"][self.time_step-8]
        self.prev9_price = self.data["close"][self.time_step-9]
        self.prev10_price = self.data["close"][self.time_step-10]
        self.close_sma = self.data["close_sma"][self.time_step]
        self.bb = self.data["bb"][self.time_step]
        self.rsi = self.data["rsi"][self.time_step]

    def get_next_state(self):
        self.update_time_step()
        next_state = [self.starting_cash - self.curr_cash,
                      self.position,
                      self.curr_price,
                      self.prev1_price,
                      self.prev2_price,
                      self.prev3_price,
                      self.prev4_price,
                      self.prev5_price,
                      self.prev6_price,
                      self.prev7_price,
                      self.prev8_price,
                      self.prev9_price,
                      self.prev10_price,
                      self.close_sma,
                      self.bb,
                      self.rsi
                     ]
        return next_state

    def execute(self, actions):
        """
        action=
        0: Buy
        1: Sell
        2: Hold
        self.position=
        0: Didn't buy any stocks
        1: Bought Stocks
        Valid Actions are:
            Buy when no stocks : action = 0 and position = 0
            Sell when stocks   : action = 1 and position = 1
            Hold when stocks   : action = 2 and position = 1
        invalid Actions are:
            Buy when stocks    : action = 0 and position = 1
            Sell when no stocks: action = 1 and position = 0
            Hold when no stocks: action = 2 and position = 0
        """
        reward = 0
        terminal = False

        ##Valid Actions
        if(actions == 0 and self.position == 0): ## Buy
            buy_value = np.round(self.stock_size * self.curr_price,3)
            if(buy_value<=self.curr_cash):
                self.curr_cash -= buy_value
                self.prev_buy = buy_value
                self.position = 1
                reward = 50
                #print("Buy")
                self.data["BuySignal"][self.time_step] = self.curr_price

        if(actions == 1 and self.position == 1 or (self.time_step == self._max_timesteps-2 and self.position==1)): ## Sell
            sell_value = np.round(self.stock_size * self.curr_price,3)
            self.curr_cash += sell_value
            self.position = 0
            profit = sell_value - self.prev_buy
            reward = 2*profit #if(profit>0) else 0
            #print("Sell")
            self.data["SellSignal"][self.time_step] = self.curr_price

        if(actions == 2 and self.position == 1): ## Hold
            curr_value = np.round(self.stock_size * self.curr_price,3)
            if(curr_value >= self.prev_buy):  ## Good to Hold
                reward = 30
                #print("Good Hold")
                self.data["HoldSignal"][self.time_step] = 2
                pass
            else:                             ## Bad to Hold
                reward = -20
                #print("Bad Hold")
                self.data["HoldSignal"][self.time_step] = 1
                pass
        ##InValid Actions
#        if(actions == 1 and self.position == 0):
#            pass
#        if(actions == 2 and self.position == 0):
#            pass
#        if(actions == 0 and self.position == 1):
#            pass

        ## Get next_state
        next_state = self.get_next_state()
        ## Terminal State Check
        if(self.time_step == self._max_timesteps-1):
            if(self.viz==True):
                self.visualize()
            terminal = True


        return next_state, terminal, reward

    def visualize(self):
        fig,ax = plt.subplots(figsize=(20,12))
        self.data["close"].reset_index().drop("DateTime",axis=1).plot(label="Price",ax = ax)
        self.data["BuySignal"].reset_index().drop("DateTime",axis=1).plot(style="*",color="Blue",label="Buy",ax = ax)
        self.data["SellSignal"].reset_index().drop("DateTime",axis=1).plot(style="*",color="Red",label="Sell",ax = ax)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Buy and Sell Signals")
        plt.legend()


