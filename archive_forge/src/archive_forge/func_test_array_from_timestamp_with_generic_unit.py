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
def test_array_from_timestamp_with_generic_unit():
    n = np.datetime64('NaT')
    x = np.datetime64('2017-01-01 01:01:01.111111111')
    y = np.datetime64('2018-11-22 12:24:48.111111111')
    with pytest.raises(pa.ArrowNotImplementedError, match='Unbound or generic datetime64 time unit'):
        pa.array([n, x, y])