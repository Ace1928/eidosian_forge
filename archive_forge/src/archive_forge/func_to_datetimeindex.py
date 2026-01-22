from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def to_datetimeindex(self, unsafe=False):
    """If possible, convert this index to a pandas.DatetimeIndex.

        Parameters
        ----------
        unsafe : bool
            Flag to turn off warning when converting from a CFTimeIndex with
            a non-standard calendar to a DatetimeIndex (default ``False``).

        Returns
        -------
        pandas.DatetimeIndex

        Raises
        ------
        ValueError
            If the CFTimeIndex contains dates that are not possible in the
            standard calendar or outside the nanosecond-precision range.

        Warns
        -----
        RuntimeWarning
            If converting from a non-standard calendar to a DatetimeIndex.

        Warnings
        --------
        Note that for non-standard calendars, this will change the calendar
        type of the index.  In that case the result of this method should be
        used with caution.

        Examples
        --------
        >>> times = xr.cftime_range("2000", periods=2, calendar="gregorian")
        >>> times
        CFTimeIndex([2000-01-01 00:00:00, 2000-01-02 00:00:00],
                    dtype='object', length=2, calendar='standard', freq=None)
        >>> times.to_datetimeindex()
        DatetimeIndex(['2000-01-01', '2000-01-02'], dtype='datetime64[ns]', freq=None)
        """
    if not self._data.size:
        return pd.DatetimeIndex([])
    nptimes = cftime_to_nptime(self)
    calendar = infer_calendar_name(self)
    if calendar not in _STANDARD_CALENDARS and (not unsafe):
        warnings.warn(f'Converting a CFTimeIndex with dates from a non-standard calendar, {calendar!r}, to a pandas.DatetimeIndex, which uses dates from the standard calendar.  This may lead to subtle errors in operations that depend on the length of time between dates.', RuntimeWarning, stacklevel=2)
    return pd.DatetimeIndex(nptimes)