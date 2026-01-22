from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_exciting(self):
    x = nonlin.excitingmixing(F, F.xin, iter=20, alpha=0.5)
    assert_(nonlin.norm(x) < 1e-05)
    assert_(nonlin.norm(F(x)) < 1e-05)