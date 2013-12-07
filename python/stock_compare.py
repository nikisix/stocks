from __future__ import print_function

import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.pyplot as plt
#from sklearn.hmm import GaussianHMM
def normalize(X, mean, sd):
    return (X - mean) / sd

print(__doc__)

###############################################################################
# Downloading the data
date1 = datetime.date(2004, 1, 1)  # start date yyyy,mm,dd
date2 = datetime.date(2012, 1, 6)  # end date
stocks = ["YHOO", "GOOG", "AAPL", "XOM"]
n_stocks = len(stocks)
quotes = {} #empty dictionary
dates = {}
close_values = {}
all_close_values = []
volumes = {}

#download quotes
for stock in stocks:
    quotes[stock] = quotes_historical_yahoo(stock, date1, date2)
    if len(quotes[stock]) == 0:
        raise SystemExit
    #unpack dates, closes and volumes
    dates[stock] = np.array([q[0] for q in quotes[stock]], dtype=int)
    close_values[stock] = np.array([q[2] for q in quotes[stock]])
    all_close_values = np.concatenate([all_close_values, close_values[stock]])
    volumes = np.array([q[5] for q in quotes[stock]])[1:]

print(dates[stock])
raise SystemExit
    
#normalize quotes - cat all values into a single array, then normalize with superset mean and sd
for stock in stocks: 
    close_values[stock] = (close_values[stock] - np.mean(all_close_values)) / np.std(all_close_values)

###################################################
#Plotting

pl.figure()
for stock in stocks: 
    pl.plot(dates[stock], close_values[stock], c=np.random.rand(3,1), label=stock, linewidth=2)
pl.xlabel("date")
pl.ylabel("closing_value")
pl.title("Stock Comparison")
pl.legend()
pl.show()

