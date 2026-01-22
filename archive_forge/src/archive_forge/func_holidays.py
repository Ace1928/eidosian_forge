from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def holidays(self, start=None, end=None, return_name: bool=False):
    """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
    if self.rules is None:
        raise Exception(f'Holiday Calendar {self.name} does not have any rules specified')
    if start is None:
        start = AbstractHolidayCalendar.start_date
    if end is None:
        end = AbstractHolidayCalendar.end_date
    start = Timestamp(start)
    end = Timestamp(end)
    if self._cache is None or start < self._cache[0] or end > self._cache[1]:
        pre_holidays = [rule.dates(start, end, return_name=True) for rule in self.rules]
        if pre_holidays:
            holidays = concat(pre_holidays)
        else:
            holidays = Series(index=DatetimeIndex([]), dtype=object)
        self._cache = (start, end, holidays.sort_index())
    holidays = self._cache[2]
    holidays = holidays[start:end]
    if return_name:
        return holidays
    else:
        return holidays.index