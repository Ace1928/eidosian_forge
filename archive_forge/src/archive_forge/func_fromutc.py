import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
def fromutc(self, dt):
    """Convert from datetime in UTC to datetime in local time"""
    if not isinstance(dt, datetime):
        raise TypeError('fromutc() requires a datetime argument')
    if dt.tzinfo is not self:
        raise ValueError('dt.tzinfo is not self')
    timestamp = self._get_local_timestamp(dt)
    num_trans = len(self._trans_utc)
    if num_trans >= 1 and timestamp < self._trans_utc[0]:
        tti = self._tti_before
        fold = 0
    elif (num_trans == 0 or timestamp > self._trans_utc[-1]) and (not isinstance(self._tz_after, _ttinfo)):
        tti, fold = self._tz_after.get_trans_info_fromutc(timestamp, dt.year)
    elif num_trans == 0:
        tti = self._tz_after
        fold = 0
    else:
        idx = bisect.bisect_right(self._trans_utc, timestamp)
        if num_trans > 1 and timestamp >= self._trans_utc[1]:
            tti_prev, tti = self._ttinfos[idx - 2:idx]
        elif timestamp > self._trans_utc[-1]:
            tti_prev = self._ttinfos[-1]
            tti = self._tz_after
        else:
            tti_prev = self._tti_before
            tti = self._ttinfos[0]
        shift = tti_prev.utcoff - tti.utcoff
        fold = shift.total_seconds() > timestamp - self._trans_utc[idx - 1]
    dt += tti.utcoff
    if fold:
        return dt.replace(fold=1)
    else:
        return dt