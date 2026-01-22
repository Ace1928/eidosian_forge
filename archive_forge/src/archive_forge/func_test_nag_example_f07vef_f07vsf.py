import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
@pytest.mark.parametrize('dtype', DTYPES)
def test_nag_example_f07vef_f07vsf(self, dtype):
    """Test real (f07vef) and complex (f07vsf) examples from NAG

        Examples available from:
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vef.html
        * https://www.nag.com/numeric/fl/nagdoc_latest/html/f07/f07vsf.html

        """
    if dtype in REAL_DTYPES:
        ab = np.array([[-4.16, 4.78, 6.32, 0.16], [-2.25, 5.86, -4.82, 0]], dtype=dtype)
        b = np.array([[-16.64, -4.16], [-13.78, -16.59], [13.1, -4.94], [-14.14, -9.96]], dtype=dtype)
        x_out = np.array([[4, 1], [-1, -3], [3, 2], [2, -2]], dtype=dtype)
    elif dtype in COMPLEX_DTYPES:
        ab = np.array([[-1.94 + 4.43j, 4.12 - 4.27j, 0.43 - 2.66j, 0.44 + 0.1j], [-3.39 + 3.44j, -1.84 + 5.52j, 1.74 - 0.04j, 0], [1.62 + 3.68j, -2.77 - 1.93j, 0, 0]], dtype=dtype)
        b = np.array([[-8.86 - 3.88j, -24.09 - 5.27j], [-15.57 - 23.41j, -57.97 + 8.14j], [-7.63 + 22.78j, 19.09 - 29.51j], [-14.74 - 2.4j, 19.17 + 21.33j]], dtype=dtype)
        x_out = np.array([[2j, 1 + 5j], [1 - 3j, -7 - 2j], [-4.001887 - 4.988417j, 3.02683 + 4.003182j], [1.996158 - 1.045105j, -6.103357 - 8.986653j]], dtype=dtype)
    else:
        raise ValueError(f'Datatype {dtype} not understood.')
    tbtrs = get_lapack_funcs('tbtrs', dtype=dtype)
    x, info = tbtrs(ab=ab, b=b, uplo='L')
    assert_equal(info, 0)
    assert_allclose(x, x_out, rtol=0, atol=1e-05)