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
def make_df_with_timestamps():
    df = pd.DataFrame({'dateTimeMs': [np.datetime64('0001-01-01 00:00', 'ms'), np.datetime64('2012-05-02 12:35', 'ms'), np.datetime64('2012-05-03 15:42', 'ms'), np.datetime64('3000-05-03 15:42', 'ms')], 'dateTimeNs': [np.datetime64('1991-01-01 00:00', 'ns'), np.datetime64('2012-05-02 12:35', 'ns'), np.datetime64('2012-05-03 15:42', 'ns'), np.datetime64('2050-05-03 15:42', 'ns')]})
    assert (df.dateTimeMs.dtype, df.dateTimeNs.dtype) == (np.dtype('O'), np.dtype('M8[ns]'))
    return df