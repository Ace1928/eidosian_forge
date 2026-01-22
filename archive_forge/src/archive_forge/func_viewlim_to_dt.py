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
def viewlim_to_dt(self):
    """Convert the view interval to datetime objects."""
    vmin, vmax = self.axis.get_view_interval()
    if vmin > vmax:
        vmin, vmax = (vmax, vmin)
    return (num2date(vmin, self.tz), num2date(vmax, self.tz))