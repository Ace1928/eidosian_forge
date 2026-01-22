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
def pade3(self):
    b = (120.0, 60.0, 12.0, 1.0)
    U = _smart_matrix_product(self.A, b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
    V = b[2] * self.A2 + b[0] * self.ident
    return (U, V)