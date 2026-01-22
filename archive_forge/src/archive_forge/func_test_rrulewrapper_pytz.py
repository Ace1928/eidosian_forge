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
@pytest.mark.pytz
def test_rrulewrapper_pytz():
    pytz = pytest.importorskip('pytz')

    def attach_tz(dt, zi):
        return zi.localize(dt)
    _test_rrulewrapper(attach_tz, pytz.timezone)