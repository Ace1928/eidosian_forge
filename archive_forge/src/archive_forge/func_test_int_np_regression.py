import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_int_np_regression():
    for t in sctypes['int'] + sctypes['uint']:
        info = np.iinfo(t)
        mn, mx = np.array([info.min, info.max], dtype=t)
        assert (mn, mx) == (int(mn), int(mx))