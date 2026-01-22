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
def test_binary_total_values_length():
    arr = pa.array([b'0000', None, b'11111', b'222222', b'3333333'], type='binary')
    large_arr = pa.array([b'0000', None, b'11111', b'222222', b'3333333'], type='large_binary')
    assert arr.total_values_length == 22
    assert arr.slice(1, 3).total_values_length == 11
    assert large_arr.total_values_length == 22
    assert large_arr.slice(1, 3).total_values_length == 11