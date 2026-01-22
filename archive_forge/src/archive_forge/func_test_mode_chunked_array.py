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
def test_mode_chunked_array():
    arr = pa.chunked_array([pa.array([1, 1, 3, 4, 3, 5], type='int64')])
    mode = pc.mode(arr)
    assert len(mode) == 1
    assert mode[0].as_py() == {'mode': 1, 'count': 2}
    mode = pc.mode(arr, n=2)
    assert len(mode) == 2
    assert mode[0].as_py() == {'mode': 1, 'count': 2}
    assert mode[1].as_py() == {'mode': 3, 'count': 2}
    arr = pa.chunked_array((), type='int64')
    assert arr.num_chunks == 0
    assert len(pc.mode(arr)) == 0