import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
@_api.deprecated('3.7')
def num2julian(n):
    """
    Convert a Matplotlib date (or sequence) to a Julian date (or sequence).

    Parameters
    ----------
    n : float or sequence of floats
        Matplotlib dates (days relative to `.get_epoch`).

    Returns
    -------
    float or sequence of floats
        Julian dates (days relative to 4713 BC Jan 1, 12:00:00).
    """
    ep = np.datetime64(get_epoch(), 'h').astype(float) / 24.0
    ep0 = np.datetime64('0000-12-31T00:00:00', 'h').astype(float) / 24.0
    dt = __getattr__('JULIAN_OFFSET') - ep0 + ep
    return np.add(n, dt)