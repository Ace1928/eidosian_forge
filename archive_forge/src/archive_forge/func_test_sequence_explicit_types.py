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
@pytest.mark.parametrize('input', [(pa.date32(), [10957, None]), (pa.date64(), [10957 * 86400000, None])])
def test_sequence_explicit_types(input):
    t, ex_values = input
    data = [datetime.date(2000, 1, 1), None]
    arr = pa.array(data, type=t)
    arr2 = pa.array(ex_values, type=t)
    for x in [arr, arr2]:
        assert len(x) == 2
        assert x.type == t
        assert x.null_count == 1
        assert x[0].as_py() == datetime.date(2000, 1, 1)
        assert x[1].as_py() is None