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
def test_array_from_naive_datetimes():
    arr = pa.array([None, datetime.datetime(2017, 4, 4, 12, 11, 10), datetime.datetime(2018, 1, 1, 0, 2, 0)])
    assert arr.type == pa.timestamp('us', tz=None)