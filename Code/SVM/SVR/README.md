# SVR/SVC for Trading !


# Approach

I used Support Vector Regression and Classification to predict if the price is going to rise in the future.

All the feature data is also standardised using StandardScaler, to avoid any scaling issues.
The features used are open,high,low,adj_close,volume,close_sma,bb,rsi - open, high,low,adj_close indicate the price levels on a particular day
and the rest are the indicator values on a particular day.
The dataset is split into 80% training and 20% test data.

### SVR
For SVR, the label for each row is the percentage increase in price after a fixed number of days (14).
At each day, we look at the open price 14 days ahead, and calculate the %age increase in price compared to the current day.
This numeric value is calculated as the labels for the data set.

### SVC
The above same setup is used for SVC also, except the labels for SVC belong to 2 classes - 0 and 1
We chose a particular threshold %age (say 1%).
For a particular day, if the above mentioned calculated increase >= threshold, it gets the label 1, otherwise 0

### Data
Tested the approach on GOG.csv, 8 years of daily price data 

### Evaluation of the above strategies
I start with a **starting cash of 1 million dollars** and execute several buy and sell orders on the test data set.
The total cash of 1 million is divided into say 10 chunks, and a chunk is used when entering a trade.
For SVR, if the predicted %age increase >= threshold (1%), then I buy shares with the 1 chunk of cash. These 
shares will be sold exactly after the fixed number of days mentioned above (14)
Similarly for SVC, if the predicted label for the day is 1, then I buy shares.
During any point of time, we will be having at max, 10 trade entries, which will each be exited (sold) after the fixed 
number of days.
After simulating all the trades, we calculate the portfolio value at the end day of the testset, which is the sum of remaining 
cash and the value of shares holding

**Performance comparison - GOOGL 2010-08-12 to 2017-08-09:** The baseline is the portfolio value, if we just bought 1 million worth of shares on the
start day, and hold it till the last day. The portfolio value is the value of all shares on the end day.

We observed that Support Vector Classification (SVC) performed better than Support Vector Regression (SVR).
The results of both models are shown below. 

|                |Final Portfolio Value          |Cumulative Return                        |
|----------------|-------------------------------|-----------------------------|
|Buy and Hold    |`1,335,584$`            |33%          |
|SVC             |`1,589,452$`            |59%            |
|SVR             |`1,319,144$`            |32%|


**SVC metrics :**

```mermaid
Accuracy of Test set = 94.85% (Percentage of correct predictions of the labels)

Precision of Test set = 91.04% (Percentage of True Positives out of Predicted Positives)
```

**SVR metrics :**

```mermaid
RMSE of Training set: 5.592886479250125
RMSE of Test set: 3.9099973736201394
```

**Plots :**

The plots of portfolio value versus time, is shown below, for both SVC and SVR.
The base line portfolio (Buy and Hold) is also shown in the plots for comparison.

![alt text](https://github.com/TarunSaranga/Tradebook/blob/master/Code/SVM/SVR/svc_.png)
![alt text](https://github.com/TarunSaranga/Tradebook/blob/master/Code/SVM/SVR/svr_.png)


**Plots with good and bad trade entries GOOGL 2015-01-01 to 2017-02-01 :**
A smaller period is selected to show the good and bad trade entries of the SVR/SVC strategies.
A green vertical line indicates the buy trade resulted in a profit after the fixed number of days when exited (sold).
Similarly a red vertical line indicates a loss on the trade. 

|                |Final Portfolio Value          |Cumulative Return                        |
|----------------|-------------------------------|-----------------------------|
|Buy and Hold    |`1,032,791$`            |3.27%          |
|SVC             |`1,097,871$`            |9.78%            |
|SVR             |`1,063,512$`            |6.35%|


**SVC metrics :**

```mermaid
Accuracy of Train set = 93.35% (Percentage of correct predictions of the labels)

Precision of Test set = 76.95% (Percentage of True Positives out of Predicted Positives)
```

**SVR metrics :**

```mermaid
RMSE of Training set: 5.2779
RMSE of Test set:  2.58843
```



![alt text](https://github.com/TarunSaranga/Tradebook/blob/master/Code/SVM/SVR/svc_trades.png)
![alt text](https://github.com/TarunSaranga/Tradebook/blob/master/Code/SVM/SVR/svr_trades.png)

**Observation :**

We can observe the Support vector trading strategy, correctly predicting a downfall, and avoiding any buy orders
during a continuous decline in price.

Also, we can see, Support Vector classification performing better than the Regression trading model.
