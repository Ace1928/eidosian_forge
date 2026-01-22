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
def test_decimal_with_None_explicit_type(self):
    series = pd.Series([decimal.Decimal('3.14'), None])
    _check_series_roundtrip(series, type_=pa.decimal128(12, 5))
    series = pd.Series([None] * 2)
    _check_series_roundtrip(series, type_=pa.decimal128(12, 5))