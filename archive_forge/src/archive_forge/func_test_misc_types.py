import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_misc_types(self):
    A = expm(np.array([[1]]))
    assert_allclose(expm(((1,),)), A)
    assert_allclose(expm([[1]]), A)
    assert_allclose(expm(matrix([[1]])), A)
    assert_allclose(expm(np.array([[1]])), A)
    assert_allclose(expm(csc_matrix([[1]])).A, A)
    B = expm(np.array([[1j]]))
    assert_allclose(expm(((1j,),)), B)
    assert_allclose(expm([[1j]]), B)
    assert_allclose(expm(matrix([[1j]])), B)
    assert_allclose(expm(csc_matrix([[1j]])).A, B)