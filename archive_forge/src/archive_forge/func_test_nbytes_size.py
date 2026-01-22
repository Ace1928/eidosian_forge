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
def test_nbytes_size():
    a = pa.chunked_array([pa.array([1, None, 3], type=pa.int16()), pa.array([4, 5, 6], type=pa.int16())])
    assert a.nbytes == 13