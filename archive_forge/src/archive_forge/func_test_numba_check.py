import importlib
from unittest.mock import Mock
import numpy as np
import pytest
from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import
def test_numba_check():
    """Test for numba_check"""
    numba = importlib.util.find_spec('numba')
    flag = numba is not None
    assert flag == numba_check()