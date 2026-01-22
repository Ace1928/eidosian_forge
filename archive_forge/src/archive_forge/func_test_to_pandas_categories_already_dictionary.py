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
def test_to_pandas_categories_already_dictionary(self):
    array = pa.array(['foo', 'foo', 'foo', 'bar']).dictionary_encode()
    table = pa.Table.from_arrays(arrays=[array], names=['col'])
    result = table.to_pandas(categories=['col'])
    assert table.to_pandas().equals(result)