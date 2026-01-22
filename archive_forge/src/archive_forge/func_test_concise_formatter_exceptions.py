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
@pytest.mark.parametrize('kwarg', ('formats', 'zero_formats', 'offset_formats'))
def test_concise_formatter_exceptions(kwarg):
    locator = mdates.AutoDateLocator()
    kwargs = {kwarg: ['', '%Y']}
    match = f'{kwarg} argument must be a list'
    with pytest.raises(ValueError, match=match):
        mdates.ConciseDateFormatter(locator, **kwargs)