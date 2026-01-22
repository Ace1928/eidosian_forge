from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_root_anderson(self):
    res = root(F, F.xin, method='anderson', options={'nit': 12, 'jac_options': {'alpha': 0.03, 'M': 5}})
    assert_(nonlin.norm(res.x) < 0.33)