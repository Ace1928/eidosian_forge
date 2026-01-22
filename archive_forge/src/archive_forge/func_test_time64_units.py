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
def test_time64_units():
    for valid_unit in ('us', 'ns'):
        ty = pa.time64(valid_unit)
        assert ty.unit == valid_unit
    for invalid_unit in ('m', 's', 'ms'):
        error_msg = 'Invalid time unit for time64: {!r}'.format(invalid_unit)
        with pytest.raises(ValueError, match=error_msg):
            pa.time64(invalid_unit)