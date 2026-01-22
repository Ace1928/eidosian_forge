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
def test_singleton_blocks_zero_copy():
    t = pa.table([pa.array(np.arange(1000, dtype=np.int64))], ['f0'])
    _check_to_pandas_memory_unchanged(t, split_blocks=True)
    prior_allocation = pa.total_allocated_bytes()
    result = t.to_pandas()
    assert result['f0']._values.flags.writeable
    assert pa.total_allocated_bytes() > prior_allocation