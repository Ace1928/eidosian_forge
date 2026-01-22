import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_list_from_numpy():
    s = pa.scalar(np.array([1, 2, 3], dtype=np.int64()))
    assert s.type == pa.list_(pa.int64())
    assert s.as_py() == [1, 2, 3]