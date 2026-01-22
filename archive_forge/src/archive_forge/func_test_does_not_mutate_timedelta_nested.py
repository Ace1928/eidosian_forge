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
def test_does_not_mutate_timedelta_nested():
    from datetime import timedelta
    timedelta_1 = [{'timedelta_1': timedelta(seconds=12, microseconds=1)}]
    timedelta_2 = [timedelta(hours=3, minutes=40, seconds=23)]
    table = pa.table({'timedelta_1': timedelta_1, 'timedelta_2': timedelta_2})
    df = table.to_pandas()
    assert df['timedelta_2'][0].to_pytimedelta() == timedelta_2[0]