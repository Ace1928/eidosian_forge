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
def expected_time_value(t):
    if unit == 's':
        return t.replace(microsecond=0)
    elif unit == 'ms':
        return t.replace(microsecond=t.microsecond // 1000 * 1000)
    else:
        return t