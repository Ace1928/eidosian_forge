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
def test_locator_set_formatter():
    """
    Test if setting the locator only will update the AutoDateFormatter to use
    the new locator.
    """
    plt.rcParams['date.autoformatter.minute'] = '%d %H:%M'
    t = [datetime.datetime(2018, 9, 30, 8, 0), datetime.datetime(2018, 9, 30, 8, 59), datetime.datetime(2018, 9, 30, 10, 30)]
    x = [2, 3, 1]
    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.xaxis.set_major_locator(mdates.MinuteLocator((0, 30)))
    fig.canvas.draw()
    ticklabels = [tl.get_text() for tl in ax.get_xticklabels()]
    expected = ['30 08:00', '30 08:30', '30 09:00', '30 09:30', '30 10:00', '30 10:30']
    assert ticklabels == expected
    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator((5, 55)))
    decoy_loc = mdates.MinuteLocator((12, 27))
    ax.xaxis.set_minor_formatter(mdates.AutoDateFormatter(decoy_loc))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator((15, 45)))
    fig.canvas.draw()
    ticklabels = [tl.get_text() for tl in ax.get_xticklabels(which='minor')]
    expected = ['30 08:15', '30 08:45', '30 09:15', '30 09:45', '30 10:15']
    assert ticklabels == expected