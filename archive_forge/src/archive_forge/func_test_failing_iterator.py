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
def test_failing_iterator():
    with pytest.raises(ZeroDivisionError):
        pa.array((1 // 0 for x in range(10)))
    with pytest.raises(ZeroDivisionError):
        pa.array((1 // 0 for x in range(10)), size=10)