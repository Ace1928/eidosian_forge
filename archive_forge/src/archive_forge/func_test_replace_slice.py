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
@pytest.mark.pandas
def test_replace_slice():
    offsets = range(-3, 4)
    arr = pa.array([None, '', 'a', 'ab', 'abc', 'abcd', 'abcde'])
    series = arr.to_pandas()
    for start in offsets:
        for stop in offsets:
            expected = series.str.slice_replace(start, stop, 'XX')
            actual = pc.binary_replace_slice(arr, start=start, stop=stop, replacement='XX')
            assert actual.tolist() == expected.tolist()
            assert pc.binary_replace_slice(arr, start, stop, 'XX') == actual
    arr = pa.array([None, '', 'π', 'πb', 'πbθ', 'πbθd', 'πbθde'])
    series = arr.to_pandas()
    for start in offsets:
        for stop in offsets:
            expected = series.str.slice_replace(start, stop, 'XX')
            actual = pc.utf8_replace_slice(arr, start=start, stop=stop, replacement='XX')
            assert actual.tolist() == expected.tolist()