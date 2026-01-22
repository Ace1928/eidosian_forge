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
@pytest.mark.parametrize('t0', [datetime.datetime(2017, 1, 1, 0, 1, 1), [datetime.datetime(2017, 1, 1, 0, 1, 1), datetime.datetime(2017, 1, 1, 1, 1, 1)], [[datetime.datetime(2017, 1, 1, 0, 1, 1), datetime.datetime(2017, 1, 1, 1, 1, 1)], [datetime.datetime(2017, 1, 1, 2, 1, 1), datetime.datetime(2017, 1, 1, 3, 1, 1)]]])
@pytest.mark.parametrize('dtype', ['datetime64[s]', 'datetime64[us]', 'datetime64[ms]', 'datetime64[ns]'])
def test_date_date2num_numpy(t0, dtype):
    time = mdates.date2num(t0)
    tnp = np.array(t0, dtype=dtype)
    nptime = mdates.date2num(tnp)
    np.testing.assert_equal(time, nptime)