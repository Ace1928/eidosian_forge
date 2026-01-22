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
def test_sequence_decimal_different_precisions():
    data = [decimal.Decimal('1234234983.183'), decimal.Decimal('80943244.234')]
    for type in [pa.decimal128, pa.decimal256]:
        arr = pa.array(data, type=type(precision=13, scale=3))
        assert arr.to_pylist() == data