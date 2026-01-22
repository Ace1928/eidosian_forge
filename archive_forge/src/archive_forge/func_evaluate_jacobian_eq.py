from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def evaluate_jacobian_eq(self, out=None):
    self._evaluate_jacobians_and_cache_if_necessary()
    if out is not None:
        if not isinstance(out, coo_matrix) or out.shape[0] != self._n_con_eq or out.shape[1] != self._n_primals or (out.nnz != self._nnz_jac_eq):
            raise RuntimeError('evaluate_jacobian_eq called with an "out" argument that is invalid. This should be a coo_matrix with shape=({},{}) and nnz={}'.format(self._n_con_eq, self._n_primals, self._nnz_jac_eq))
        self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=out.data)
        return out
    else:
        self._cached_jac_full.data.compress(self._nz_con_full_eq_mask, out=self._cached_jac_eq.data)
        return self._cached_jac_eq.copy()