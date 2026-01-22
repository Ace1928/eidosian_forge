from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_filter_errors():
    arr = pa.chunked_array([['a', None], ['c', 'd', 'e']])
    batch = pa.record_batch([pa.array(['a', None, 'c', 'd', 'e'])], names=["a'"])
    table = pa.table([pa.array(['a', None, 'c', 'd', 'e'])], names=['a'])
    for obj in [arr, batch, table]:
        mask = pa.array([0, 1, 0, 1, 0])
        with pytest.raises(NotImplementedError):
            obj.filter(mask)
        mask = pa.array([True, False, True])
        with pytest.raises(pa.ArrowInvalid, match='must all be the same length'):
            obj.filter(mask)
    scalar = pa.scalar(True)
    for filt in [batch, table, scalar]:
        with pytest.raises(TypeError):
            table.filter(filt)