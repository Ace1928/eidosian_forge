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
def test_rangeindex_doesnt_warn(self):
    df = pd.DataFrame(np.random.randn(4, 2), columns=['a', 'b'])
    with warnings.catch_warnings():
        warnings.simplefilter(action='error')
        warnings.filterwarnings('ignore', 'make_block is deprecated', DeprecationWarning)
        _check_pandas_roundtrip(df, preserve_index=True)