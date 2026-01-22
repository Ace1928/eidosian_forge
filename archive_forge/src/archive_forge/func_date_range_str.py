from statsmodels.compat.python import asstr, lmap, lrange, lzip
import datetime
import re
import numpy as np
from pandas import to_datetime
def date_range_str(start, end=None, length=None):
    """
    Returns a list of abbreviated date strings.

    Parameters
    ----------
    start : str
        The first abbreviated date, for instance, '1965q1' or '1965m1'
    end : str, optional
        The last abbreviated date if length is None.
    length : int, optional
        The length of the returned array of end is None.

    Returns
    -------
    date_range : list
        List of strings
    """
    flags = re.IGNORECASE | re.VERBOSE
    start = start.lower()
    if re.search(_m_pattern, start, flags):
        annual_freq = 12
        split = 'm'
    elif re.search(_q_pattern, start, flags):
        annual_freq = 4
        split = 'q'
    elif re.search(_y_pattern, start, flags):
        annual_freq = 1
        start += 'a1'
        if end:
            end += 'a1'
        split = 'a'
    else:
        raise ValueError('Date %s not understood' % start)
    yr1, offset1 = lmap(int, start.replace(':', '').split(split))
    if end is not None:
        end = end.lower()
        yr2, offset2 = lmap(int, end.replace(':', '').split(split))
    else:
        if not length:
            raise ValueError('length must be provided if end is None')
        yr2 = yr1 + length // annual_freq
        offset2 = length % annual_freq + (offset1 - 1)
    years = [str(yr) for yr in np.repeat(lrange(yr1 + 1, yr2), annual_freq)]
    years = [str(yr1)] * (annual_freq + 1 - offset1) + years
    years = years + [str(yr2)] * offset2
    if split != 'a':
        offset = np.tile(np.arange(1, annual_freq + 1), yr2 - yr1 - 1).astype('S2')
        offset = np.r_[np.arange(offset1, annual_freq + 1).astype('S2'), offset]
        offset = np.r_[offset, np.arange(1, offset2 + 1).astype('S2')]
        date_arr_range = [''.join([i, split, asstr(j)]) for i, j in zip(years, offset)]
    else:
        date_arr_range = years
    return date_arr_range