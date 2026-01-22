import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def expected_datetime_value(dt):
    if dt is None:
        return None
    if unit == 's':
        dt = dt.replace(microsecond=0)
    elif unit == 'ms':
        dt = dt.replace(microsecond=dt.microsecond // 1000 * 1000)
    if timezone is None:
        return dt.replace(tzinfo=None)
    else:
        return dt.astimezone(pytz.timezone(timezone))