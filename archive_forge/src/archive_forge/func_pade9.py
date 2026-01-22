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
def pade9(self):
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0)
    U = _smart_matrix_product(self.A, b[9] * self.A8 + b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
    V = b[8] * self.A8 + b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
    return (U, V)