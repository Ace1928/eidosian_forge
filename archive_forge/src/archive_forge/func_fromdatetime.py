from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@classmethod
def fromdatetime(cls, dt: Union[datetime.datetime, datetime.date, datetime.time], year: Optional[int]=None) -> 'AbstractDateTime':
    """
        Creates an XSD date/time instance from a datetime.datetime/date/time instance.

        :param dt: the datetime, date or time instance that stores the XSD Date/Time value.
        :param year: if an year is provided the created instance refers to it and the         possibly present *dt.year* part is ignored.
        :return: an AbstractDateTime concrete subclass instance.
        """
    if not isinstance(dt, (datetime.datetime, datetime.date, datetime.time)):
        raise TypeError('1st argument has an invalid type %r' % type(dt))
    elif year is not None and (not isinstance(year, int)):
        raise TypeError('2nd argument has an invalid type %r' % type(year))
    kwargs = {k: getattr(dt, k) for k in cls.pattern.groupindex.keys() if hasattr(dt, k)}
    if year is not None:
        kwargs['year'] = year
    return cls(**kwargs)