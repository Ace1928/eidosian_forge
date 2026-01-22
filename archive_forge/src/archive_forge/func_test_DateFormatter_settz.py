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
def test_DateFormatter_settz():
    time = mdates.date2num(datetime.datetime(2011, 1, 1, 0, 0, tzinfo=mdates.UTC))
    formatter = mdates.DateFormatter('%Y-%b-%d %H:%M')
    assert formatter(time) == '2011-Jan-01 00:00'
    formatter.set_tzinfo('Pacific/Kiritimati')
    assert formatter(time) == '2011-Jan-01 14:00'