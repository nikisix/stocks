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

###################################################
#Plotting

fig, (ax0, ax1) = plt.subplots(nrows=2)

ax0.plot(yhoo_dates[1:20], yhoo_close_v[1:20])
ax0.set_title('yahoo')

ax1.plot(goog_dates[1:20], goog_close_v[1:20])
ax1.set_title('goog')

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.5)
plt.show()
