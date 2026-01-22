import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
def test_numba_utils():
    """Test for class Numba."""
    flag = Numba.numba_flag
    assert flag == numba_check()
    Numba.disable_numba()
    val = Numba.numba_flag
    assert not val
    Numba.enable_numba()
    val = Numba.numba_flag
    assert val
    assert flag == Numba.numba_flag