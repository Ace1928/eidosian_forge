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
def test_burkardt_11(self):
    A = np.array([[29.87942128909879, 0.7815750847907159, -2.289519314033932], [0.7815750847907159, 25.72656945571064, 8.680737820540138], [-2.289519314033932, 8.680737820540138, 34.39400925519054]], dtype=float)
    assert_allclose(scipy.linalg.eigvalsh(A), (20, 30, 40))
    desired = np.array([[5496313853692378.0, -1.823188097200898e+16, -3.047577080858001e+16], [-1.823188097200899e+16, 6.060522870222108e+16, 1.012918429302482e+17], [-3.047577080858001e+16, 1.012918429302482e+17, 1.692944112408493e+17]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)