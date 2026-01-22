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
def test_take_on_chunked_array():
    arr = pa.chunked_array([['a', 'b', 'c', 'd', 'e'], ['f', 'g', 'h', 'i', 'j']])
    indices = np.array([0, 5, 1, 6, 9, 2])
    result = arr.take(indices)
    expected = pa.chunked_array([['a', 'f', 'b', 'g', 'j', 'c']])
    assert result.equals(expected)
    indices = pa.chunked_array([[1], [9, 2]])
    result = arr.take(indices)
    expected = pa.chunked_array([['b'], ['j', 'c']])
    assert result.equals(expected)