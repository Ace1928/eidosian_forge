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
def test_change_converter():
    plt.rcParams['date.converter'] = 'concise'
    dates = np.arange('2020-01-01', '2020-05-01', dtype='datetime64[D]')
    fig, ax = plt.subplots()
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == 'Jan'
    assert ax.get_xticklabels()[1].get_text() == '15'
    plt.rcParams['date.converter'] = 'auto'
    fig, ax = plt.subplots()
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    assert ax.get_xticklabels()[0].get_text() == 'Jan 01 2020'
    assert ax.get_xticklabels()[1].get_text() == 'Jan 15 2020'
    with pytest.raises(ValueError):
        plt.rcParams['date.converter'] = 'boo'