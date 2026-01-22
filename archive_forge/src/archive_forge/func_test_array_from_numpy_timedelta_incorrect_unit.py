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
def test_array_from_numpy_timedelta_incorrect_unit():
    td = np.timedelta64(1)
    for data in [[td], np.array([td])]:
        with pytest.raises(NotImplementedError):
            pa.array(data)
    td = np.timedelta64(1, 'M')
    for data in [[td], np.array([td])]:
        with pytest.raises(NotImplementedError):
            pa.array(data)