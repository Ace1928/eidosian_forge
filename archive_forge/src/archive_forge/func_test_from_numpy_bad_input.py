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
def test_from_numpy_bad_input(self):
    ty = pa.struct([pa.field('x', pa.int32()), pa.field('y', pa.bool_())])
    dt = np.dtype([('x', np.int32), ('z', np.bool_)])
    data = np.array([], dtype=dt)
    with pytest.raises(ValueError, match="Missing field 'y'"):
        pa.array(data, type=ty)
    data = np.int32([])
    with pytest.raises(TypeError, match='Expected struct array'):
        pa.array(data, type=ty)