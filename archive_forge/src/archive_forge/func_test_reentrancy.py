import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
import pytest
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_reentrancy(solver):
    reentrant = [lgmres, minres, gcrotmk, tfqmr]
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, '.*called without specifying.*')
        _check_reentrancy(solver, solver in reentrant)