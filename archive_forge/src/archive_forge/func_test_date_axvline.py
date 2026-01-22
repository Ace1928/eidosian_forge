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
@image_comparison(['date_axvline.png'])
def test_date_axvline():
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 21)
    fig, ax = plt.subplots()
    ax.axvline(t0, color='red', lw=3)
    ax.set_xlim(t0 - datetime.timedelta(days=5), tf + datetime.timedelta(days=5))
    fig.autofmt_xdate()