import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_non_natural_stride(self):
    """
        ARROW-2172: converting from a Numpy array with a stride that's
        not a multiple of itemsize.
        """
    dtype = np.dtype([('x', np.int32), ('y', np.int16)])
    data = np.array([(42, -1), (-43, 2)], dtype=dtype)
    assert data.strides == (6,)
    arr = pa.array(data['x'], type=pa.int32())
    assert arr.to_pylist() == [42, -43]
    arr = pa.array(data['y'], type=pa.int16())
    assert arr.to_pylist() == [-1, 2]