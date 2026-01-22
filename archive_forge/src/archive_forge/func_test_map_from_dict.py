from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_map_from_dict():
    tup_arr = pa.array([[('a', 1), ('b', 2)], [('c', 3)]], pa.map_(pa.string(), pa.int64()))
    dict_arr = pa.array([{'a': 1, 'b': 2}, {'c': 3}], pa.map_(pa.string(), pa.int64()))
    assert tup_arr.equals(dict_arr)