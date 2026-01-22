import calendar
import re
import sys
from datetime import date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time
from typing import (
from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta
from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES
from arrow.locales import TimeFrameLiteral
@classmethod
def fromdate(cls, date: date, tzinfo: Optional[TZ_EXPR]=None) -> 'Arrow':
    """Constructs an :class:`Arrow <arrow.arrow.Arrow>` object from a ``date`` and optional
        replacement timezone.  All time values are set to 0.

        :param date: the ``date``
        :param tzinfo: (optional) A :ref:`timezone expression <tz-expr>`.  Defaults to UTC.

        """
    if tzinfo is None:
        tzinfo = dateutil_tz.tzutc()
    return cls(date.year, date.month, date.day, tzinfo=tzinfo)