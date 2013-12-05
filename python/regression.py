from __future__ import print_function

import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.pyplot as plt
#from sklearn.hmm import GaussianHMM

print(__doc__)

###############################################################################
# Downloading the data
date1 = datetime.date(1995, 1, 1)  # start date
date2 = datetime.date(2012, 1, 6)  # end date

# for stock in stocks
# get goog quotes from yahoo finance
goog_quotes = quotes_historical_yahoo("GOOG", date1, date2)
if len(goog_quotes) == 0:
    raise SystemExit

# get yahoo quotes from yahoo finance
yhoo_quotes = quotes_historical_yahoo("YHOO", date1, date2)
if len(yhoo_quotes) == 0:
    raise SystemExit

# unpack goog quotes
goog_dates = np.array([q[0] for q in goog_quotes], dtype=int)
goog_close_v = np.array([q[2] for q in goog_quotes])
goog_volume = np.array([q[5] for q in goog_quotes])[1:]

# unpack yhoo quotes
yhoo_dates = np.array([q[0] for q in yhoo_quotes], dtype=int)
yhoo_close_v = np.array([q[2] for q in yhoo_quotes])
yhoo_volume = np.array([q[5] for q in yhoo_quotes])[1:]

# regression tree plot
X = goog_dates[1:20]
y = goog_close_v[1:20]

# Fit regression model
from sklearn.tree import DecisionTreeRegressor

clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=5)
clf_1.fit(X, y)
clf_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)


###################################################
# Plotting

pl.figure()
pl.scatter(X, y, c="k", label="data")
pl.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
pl.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression")
pl.legend()
pl.show()
