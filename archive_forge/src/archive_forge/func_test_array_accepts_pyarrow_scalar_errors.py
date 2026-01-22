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
@parametrize_with_collections_types
def test_array_accepts_pyarrow_scalar_errors(seq):
    sequence = seq([pa.scalar(1), pa.scalar('a'), pa.scalar(3.0)])
    with pytest.raises(pa.ArrowInvalid, match='cannot mix scalars with different types'):
        pa.array(sequence)
    sequence = seq([1, pa.scalar('a'), None])
    with pytest.raises(pa.ArrowInvalid, match='pyarrow scalars cannot be mixed with other Python scalar values currently'):
        pa.array(sequence)
    sequence = seq([np.float16('0.1'), pa.scalar('a'), None])
    with pytest.raises(pa.ArrowInvalid, match='pyarrow scalars cannot be mixed with other Python scalar values currently'):
        pa.array(sequence)
    sequence = seq([pa.scalar('a'), np.float16('0.1'), None])
    with pytest.raises(pa.ArrowInvalid, match='pyarrow scalars cannot be mixed with other Python scalar values currently'):
        pa.array(sequence)
    with pytest.raises(pa.ArrowInvalid, match='Cannot append scalar of type string to builder for type int32'):
        pa.array([pa.scalar('a')], type=pa.int32())
    with pytest.raises(pa.ArrowInvalid, match='Cannot append scalar of type int64 to builder for type null'):
        pa.array([pa.scalar(1)], type=pa.null())