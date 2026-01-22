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
def test_sequence_integer_nested_np_nan(seq, np_scalar_pa_type):
    _, pa_type = np_scalar_pa_type
    with pytest.raises(ValueError):
        pa.array(seq([[np.nan]]), type=pa.list_(pa_type), from_pandas=False)
    arr = pa.array(seq([[np.nan]]), type=pa.list_(pa_type), from_pandas=True)
    expected = [[None]]
    assert len(arr) == 1
    assert arr.null_count == 0
    assert arr.type == pa.list_(pa_type)
    assert arr.to_pylist() == expected