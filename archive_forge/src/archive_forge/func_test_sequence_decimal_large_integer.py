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
def test_sequence_decimal_large_integer():
    data = [decimal.Decimal('-394029506937548693.42983'), decimal.Decimal('32358695912932.01033')]
    for type in [pa.decimal128, pa.decimal256]:
        arr = pa.array(data, type=type(precision=23, scale=5))
        assert arr.to_pylist() == data