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
@pytest.mark.parametrize('arrow_type', numerical_arrow_types)
def test_fill_null_chunked_array(arrow_type):
    fill_value = pa.scalar(5, type=arrow_type)
    arr = pa.chunked_array([pa.array([None, 2, 3, 4], type=arrow_type)])
    result = arr.fill_null(fill_value)
    expected = pa.chunked_array([pa.array([5, 2, 3, 4], type=arrow_type)])
    assert result.equals(expected)
    arr = pa.chunked_array([pa.array([1, 2], type=arrow_type), pa.array([], type=arrow_type), pa.array([None, 4], type=arrow_type)])
    expected = pa.chunked_array([pa.array([1, 2], type=arrow_type), pa.array([], type=arrow_type), pa.array([5, 4], type=arrow_type)])
    result = arr.fill_null(fill_value)
    assert result.equals(expected)
    result = arr.fill_null(5)
    assert result.equals(expected)
    result = arr.fill_null(pa.scalar(5, type='int8'))
    assert result.equals(expected)