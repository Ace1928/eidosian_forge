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
def test_sequence_decimal_infer():
    for data, typ in [(decimal.Decimal('1.234'), pa.decimal128(4, 3)), (decimal.Decimal('12300'), pa.decimal128(5, 0)), (decimal.Decimal('12300.0'), pa.decimal128(6, 1)), (decimal.Decimal('1.23E+4'), pa.decimal128(5, 0)), (decimal.Decimal('123E+2'), pa.decimal128(5, 0)), (decimal.Decimal('123E+4'), pa.decimal128(7, 0)), (decimal.Decimal('0.0123'), pa.decimal128(4, 4)), (decimal.Decimal('0.01230'), pa.decimal128(5, 5)), (decimal.Decimal('1.230E-2'), pa.decimal128(5, 5))]:
        assert pa.infer_type([data]) == typ
        arr = pa.array([data])
        assert arr.type == typ
        assert arr.to_pylist()[0] == data