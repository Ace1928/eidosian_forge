from __future__ import print_function, division, absolute_import
from collections import OrderedDict
from datetime import datetime, date, time, timedelta
from itertools import chain
import re
from textwrap import dedent
from types import MappingProxyType
from warnings import warn
from dateutil.parser import parse as dateparse
import numpy as np
from .dispatch import dispatch
from .coretypes import (int32, int64, float64, bool_, complex128, datetime_,
from .predicates import isdimension, isrecord
from .internal_utils import _toposort, groupby
from .util import subclasses
def deltaparse(x):
    """Naive timedelta string parser

    Examples
    --------
    >>> td = '1 day'
    >>> deltaparse(td)
    numpy.timedelta64(1,'D')
    >>> deltaparse('1.2 days')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: floating point timedelta value not supported
    """
    value, unit = re.split('\\s+', x.strip())
    value = float(value)
    if not value.is_integer():
        raise ValueError('floating point timedelta values not supported')
    return np.timedelta64(int(value), TimeDelta(unit=unit).unit)