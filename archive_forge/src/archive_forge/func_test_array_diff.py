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
def test_array_diff():
    arr1 = pa.array(['foo'], type=pa.utf8())
    arr2 = pa.array(['foo', 'bar', None], type=pa.utf8())
    arr3 = pa.array([1, 2, 3])
    arr4 = pa.array([[], [1], None], type=pa.list_(pa.int64()))
    assert arr1.diff(arr1) == ''
    assert arr1.diff(arr2) == '\n@@ -1, +1 @@\n+"bar"\n+null\n'
    assert arr1.diff(arr3).strip() == '# Array types differed: string vs int64'
    assert arr1.diff(arr3).strip() == '# Array types differed: string vs int64'
    assert arr1.diff(arr4).strip() == '# Array types differed: string vs list<item: int64>'