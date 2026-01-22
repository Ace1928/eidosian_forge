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
def test_datestr2num():
    assert mdates.datestr2num('2022-01-10') == 19002.0
    dt = datetime.date(year=2022, month=1, day=10)
    assert mdates.datestr2num('2022-01', default=dt) == 19002.0
    assert np.all(mdates.datestr2num(['2022-01', '2022-02'], default=dt) == np.array([19002.0, 19033.0]))
    assert mdates.datestr2num([]).size == 0
    assert mdates.datestr2num([], datetime.date(year=2022, month=1, day=10)).size == 0