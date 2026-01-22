import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
@property
def d8_loose(self):
    if self.use_exact_onenorm:
        return self.d8_tight
    if self._d8_exact is not None:
        return self._d8_exact
    else:
        if self._d8_approx is None:
            self._d8_approx = _onenormest_matrix_power(self.A4, 2, structure=self.structure) ** (1 / 8.0)
        return self._d8_approx