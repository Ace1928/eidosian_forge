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
def test_getitem_NULL():
    arr = pa.array([1, None, 2])
    assert arr[1].as_py() is None
    assert arr[1].is_valid is False
    assert isinstance(arr[1], pa.Int64Scalar)