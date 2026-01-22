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
def test_recordbatch_table_pass_name_to_pandas():
    rb = pa.record_batch([pa.array([1, 2, 3, 4])], names=['a0'])
    t = pa.table([pa.array([1, 2, 3, 4])], names=['a0'])
    assert rb[0].to_pandas().name == 'a0'
    assert t[0].to_pandas().name == 'a0'