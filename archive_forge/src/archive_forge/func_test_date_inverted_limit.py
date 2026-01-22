import datetime
import dateutil.tz
import dateutil.rrule
import functools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import rc_context, style
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.ticker as mticker
@image_comparison(['date_inverted_limit.png'])
def test_date_inverted_limit():
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 31)
    fig, ax = plt.subplots()
    ax.axhline(t0, color='blue', lw=3)
    ax.set_ylim(t0 - datetime.timedelta(days=5), tf + datetime.timedelta(days=5))
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.25)