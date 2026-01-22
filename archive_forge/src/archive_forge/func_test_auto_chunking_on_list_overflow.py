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
@pytest.mark.large_memory
def test_auto_chunking_on_list_overflow(self):
    n = 2 ** 21
    df = pd.DataFrame.from_dict({'a': list(np.zeros((n, 2 ** 10), dtype='uint8')), 'b': range(n)})
    table = pa.Table.from_pandas(df)
    table.validate(full=True)
    column_a = table[0]
    assert column_a.num_chunks == 2
    assert len(column_a.chunk(0)) == 2 ** 21 - 1
    assert len(column_a.chunk(1)) == 1