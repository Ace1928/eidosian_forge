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
def test_dictionary_indices_boundscheck(self):
    indices = [[0, 1], [0, -1]]
    for inds in indices:
        arr = pa.DictionaryArray.from_arrays(inds, ['a'], safe=False)
        batch = pa.RecordBatch.from_arrays([arr], ['foo'])
        table = pa.Table.from_batches([batch, batch, batch])
        with pytest.raises(IndexError):
            arr.to_pandas()
        with pytest.raises(IndexError):
            table.to_pandas()