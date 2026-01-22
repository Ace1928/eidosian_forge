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
def test_sequence_decimal_given_type():
    for data, typs, wrong_typs in [(decimal.Decimal('1.234'), [pa.decimal128(4, 3), pa.decimal128(5, 3), pa.decimal128(5, 4)], [pa.decimal128(4, 2), pa.decimal128(4, 4)]), (decimal.Decimal('12300'), [pa.decimal128(5, 0), pa.decimal128(6, 0), pa.decimal128(3, -2)], [pa.decimal128(4, 0), pa.decimal128(3, -3)]), (decimal.Decimal('1.23E+4'), [pa.decimal128(5, 0), pa.decimal128(6, 0), pa.decimal128(3, -2)], [pa.decimal128(4, 0), pa.decimal128(3, -3)])]:
        for typ in typs:
            arr = pa.array([data], type=typ)
            assert arr.type == typ
            assert arr.to_pylist()[0] == data
        for typ in wrong_typs:
            with pytest.raises(ValueError):
                pa.array([data], type=typ)