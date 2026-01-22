from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_timestamp_print():
    for unit in ('s', 'ms', 'us', 'ns'):
        for tz in ('UTC', 'Europe/Paris', 'Pacific/Marquesas', 'Mars/Mariner_Valley', '-00:42', '+42:00'):
            ty = pa.timestamp(unit, tz=tz)
            arr = pa.array([0], ty)
            assert 'Z' in str(arr)
        arr = pa.array([0], pa.timestamp(unit))
        assert 'Z' not in str(arr)