from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime
def dates_from_str(dates):
    """
    Turns a sequence of date strings and returns a list of datetime.

    Parameters
    ----------
    dates : array_like
        A sequence of abbreviated dates as string. For instance,
        '1996m1' or '1996Q1'. The datetime dates are at the end of the
        period.

    Returns
    -------
    date_list : ndarray
        A list of datetime types.
    """
    return lmap(date_parser, dates)