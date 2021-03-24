import yfinance as yf
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import re, glob, pickle, time
from datetime import datetime
from math import isnan


# download 1 year of SP100 data (NOTE: this is SP100 as of January 1, 2021)
def download_1y_SP100(filename):
    sp100 = pd.read_excel('sp100.xlsx')
    sp100_tickers = list(sp100['Symbol'].values)
    sp100_str = ' '.join(sp100_tickers)
    data = yf.download(sp100_str, start="2018-02-27", end= "2021-02-27",interval='1d')
    pickle.dump(data,open('{}.pickle'.format(filename),'wb'))

# download data if not already in local file
filename = '1ySP100'
if '{}.pickle'.format(filename) not in glob.glob('*.pickle'):
    download_1y_SP100(filename)
    
# load in data
data = pickle.load(open('{}.pickle'.format(filename),'rb'))['Adj Close']


# index portfolio for single ticker
ticker = 'PG'
returns = np.log(data/data.shift(periods=1))[1:].fillna(0)
ticker_returns = returns.pop(ticker)
to_test = 50 # number of days to test model on (last n days of data)

# creating regularization matrix lambda*Ix = 0
'''

lambda = C^2

    |C 0 0|   
L = |0 C 0| 
    |0 0 C|    

To solve:

     |0|
Lx = |0| 
     |0|

This will penalize abnormally large weights. Intuition of why this may be helpful in fitting the model:

Each stock has idiosycratic risk -- independent and not predictable, which is usually experienced in abnormally large movements (bad earnings report, FDA approval, etc.).
In these instances, the large movement will heavily influence the least squares model and will search for another/other stock(s) which -- by chance -- had similar movements and overweight it(them).

The new least squares problem becomes:

|A|   |b|
|L|x =|0|

'''

plt.figure(figsize=(16,9))
for c in [0,5,10,25,50,100]:
    C = c*10**(-2)
    L = C*np.identity(len(returns.columns))
    zeros = np.zeros(len(returns.columns))

    # creating training set -- will overfit if we use least squares on entire timeframe
    training_returns = np.array(returns[:-to_test])
    training_ticker = np.array(ticker_returns[:-to_test])

    A_L = np.concatenate((training_returns,L),axis=0)
    b_0 = np.concatenate((training_ticker,zeros))

    # solving Ax=b using least squares method
    x = np.linalg.lstsq(A_L, b_0,rcond=None)[0]
    x = x/sum(x)

    # testing portfolio
    portfolio = np.array(returns) @ x

    '''# plotting performance on daily returns
    plt.figure(figsize=(16,9))
    days = len(portfolio)
    plt.plot(range(days),np.array(ticker_returns),linewidth=.5,color='black',label = ticker)
    plt.plot(range(days-to_test+1),portfolio[:-to_test+1],linewidth=3,alpha=.5,color='gray', label='Training set (least squares)')
    plt.plot(range(days-to_test,days),portfolio[-to_test:],linewidth=1,color='green', label='Test portfolio')
    plt.legend()
    plt.show()'''

    ###
    # Now plotting actual returns with first day standardized to 100
    ##
    standardized_returns = (data.div(data.iloc[0])*100).fillna(100)
    ticker_cumulative_returns = standardized_returns.pop(ticker)
    cumulative_portfolio = np.array(standardized_returns) @ x

    # plotting performance on cumulative returns
    #plt.figure(figsize=(16,9))
    days = len(cumulative_portfolio)
    plt.plot(range(days),np.array(ticker_cumulative_returns),linewidth=.5,color='black')
    plt.plot(range(days-to_test+1),cumulative_portfolio[:-to_test+1],linewidth=4,alpha=(c/3+10)/50,color='gray',label="{}*10^-2".format(c))
    plt.plot(range(days-to_test,days),cumulative_portfolio[-to_test:],linewidth=1,alpha=(c/3+10)/50,color='green')

    #plt.show()

    '''plt.figure(figsize=(16,9))
    short_test = np.array(ticker_cumulative_returns)[-to_test:]
    standardize_ratio = ticker_cumulative_returns[-to_test]/cumulative_portfolio[-to_test]
    plt.plot(range(to_test),np.array(ticker_cumulative_returns)[-to_test:],linewidth=.5,color='black',label=ticker)
    plt.plot(range(to_test),cumulative_portfolio[-to_test:]*standardize_ratio,linewidth=1,color='green',label='portfolio')
    plt.title('{} over previous {} days'.format(ticker,to_test))
    plt.legend()
    plt.show()'''
plt.legend()
plt.show()




