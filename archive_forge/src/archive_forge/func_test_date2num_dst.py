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
def test_date2num_dst():

    class dt_tzaware(datetime.datetime):
        """
        This bug specifically occurs because of the normalization behavior of
        pandas Timestamp objects, so in order to replicate it, we need a
        datetime-like object that applies timezone normalization after
        subtraction.
        """

        def __sub__(self, other):
            r = super().__sub__(other)
            tzinfo = getattr(r, 'tzinfo', None)
            if tzinfo is not None:
                localizer = getattr(tzinfo, 'normalize', None)
                if localizer is not None:
                    r = tzinfo.normalize(r)
            if isinstance(r, datetime.datetime):
                r = self.mk_tzaware(r)
            return r

        def __add__(self, other):
            return self.mk_tzaware(super().__add__(other))

        def astimezone(self, tzinfo):
            dt = super().astimezone(tzinfo)
            return self.mk_tzaware(dt)

        @classmethod
        def mk_tzaware(cls, datetime_obj):
            kwargs = {}
            attrs = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo')
            for attr in attrs:
                val = getattr(datetime_obj, attr, None)
                if val is not None:
                    kwargs[attr] = val
            return cls(**kwargs)

    def date_range(start, freq, periods):
        dtstart = dt_tzaware.mk_tzaware(start)
        return [dtstart + i * freq for i in range(periods)]

    def tz_convert(dt_list, tzinfo):
        return [d.astimezone(tzinfo) for d in dt_list]
    _test_date2num_dst(date_range, tz_convert)