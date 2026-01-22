import importlib
import numpy as np
import pytest
from ...data import load_arviz_data
from ...rcparams import rcParams
from ...stats import bfmi, mcse, rhat
from ...stats.diagnostics import _mc_error, ks_summary
from ...utils import Numba
from ..helpers import running_on_ci
from .test_diagnostics import data  # pylint: disable=unused-import
@pytest.mark.parametrize('batches', (1, 20))
@pytest.mark.parametrize('circular', (True, False))
def test_mcse_error_numba(batches, circular):
    """Numba test for mcse_error."""
    data = np.random.randn(100, 100)
    state = Numba.numba_flag
    Numba.disable_numba()
    non_numba = _mc_error(data, batches=batches, circular=circular)
    Numba.enable_numba()
    with_numba = _mc_error(data, batches=batches, circular=circular)
    assert np.allclose(non_numba, with_numba)
    assert state == Numba.numba_flag