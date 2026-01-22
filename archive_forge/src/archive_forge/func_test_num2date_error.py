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
@pytest.mark.parametrize('val', (-1000000, 10000000))
def test_num2date_error(val):
    with pytest.raises(ValueError, match=f'Date ordinal {val} converts'):
        mdates.num2date(val)