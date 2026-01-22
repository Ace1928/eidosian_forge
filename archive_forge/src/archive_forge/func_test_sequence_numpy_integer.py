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
@parametrize_with_sequence_types
@pytest.mark.parametrize('np_scalar_pa_type', int_type_pairs)
def test_sequence_numpy_integer(seq, np_scalar_pa_type):
    np_scalar, pa_type = np_scalar_pa_type
    expected = [np_scalar(1), None, np_scalar(3), None, np_scalar(np.iinfo(np_scalar).min), np_scalar(np.iinfo(np_scalar).max)]
    arr = pa.array(seq(expected), type=pa_type)
    assert len(arr) == 6
    assert arr.null_count == 2
    assert arr.type == pa_type
    assert arr.to_pylist() == expected