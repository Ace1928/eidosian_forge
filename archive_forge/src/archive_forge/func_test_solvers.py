import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_square_sparrays
@pytest.mark.parametrize('solver', ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'lgmres', 'minres', 'qmr', 'gcrotmk', 'tfqmr'])
def test_solvers(B, solver):
    if solver == 'minres':
        kwargs = {}
    else:
        kwargs = {'atol': 1e-05}
    x, info = getattr(spla, solver)(B, np.array([1, 2]), **kwargs)
    assert info >= 0
    npt.assert_allclose(x, [1, 1], atol=0.1)