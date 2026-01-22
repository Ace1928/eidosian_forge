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
def test_boolean_true_count_false_count():
    arr = pa.array([True, True, None, False, None, True] * 1000)
    assert arr.true_count == 3000
    assert arr.false_count == 1000