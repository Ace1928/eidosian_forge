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
@pytest.mark.parametrize('x, tdelta', [(1, datetime.timedelta(days=1)), ([1, 1.5], [datetime.timedelta(days=1), datetime.timedelta(days=1.5)])])
def test_num2timedelta(x, tdelta):
    dt = mdates.num2timedelta(x)
    assert dt == tdelta