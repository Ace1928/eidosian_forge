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
def test_warn_notintervals():
    dates = np.arange('2001-01-10', '2001-03-04', dtype='datetime64[D]')
    locator = mdates.AutoDateLocator(interval_multiples=False)
    locator.intervald[3] = [2]
    locator.create_dummy_axis()
    locator.axis.set_view_interval(mdates.date2num(dates[0]), mdates.date2num(dates[-1]))
    with pytest.warns(UserWarning, match='AutoDateLocator was unable'):
        locs = locator()