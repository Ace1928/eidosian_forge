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
def test_to_pandas_split_blocks():
    t = pa.table([pa.array([1, 2, 3, 4, 5], type='i1'), pa.array([1, 2, 3, 4, 5], type='i4'), pa.array([1, 2, 3, 4, 5], type='i8'), pa.array([1, 2, 3, 4, 5], type='f4'), pa.array([1, 2, 3, 4, 5], type='f8'), pa.array([1, 2, 3, 4, 5], type='f8'), pa.array([1, 2, 3, 4, 5], type='f8'), pa.array([1, 2, 3, 4, 5], type='f8')], ['f{}'.format(i) for i in range(8)])
    _check_blocks_created(t, 8)
    _check_to_pandas_memory_unchanged(t, split_blocks=True)