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
def test_sequence_decimal_from_integers():
    data = [0, 1, -39402950693754869342983]
    expected = [decimal.Decimal(x) for x in data]
    for type in [pa.decimal128, pa.decimal256]:
        arr = pa.array(data, type=type(precision=28, scale=5))
        assert arr.to_pylist() == expected