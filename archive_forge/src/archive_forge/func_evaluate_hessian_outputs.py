import pyomo.environ as pyo
import numpy as np
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
def evaluate_hessian_outputs(self):
    sv = self._input_values[0]
    caf = self._input_values[1]
    ca = self._input_values[2]
    cb = self._input_values[3]
    cc = self._input_values[4]
    cd = self._input_values[5]
    k1 = 5 / 6
    k2 = 5 / 3
    k3 = 1 / 6000
    lam = self._output_con_mult_values
    nnz = 9
    irow = np.zeros(nnz, dtype=np.int64)
    jcol = np.zeros(nnz, dtype=np.int64)
    data = np.zeros(nnz, dtype=np.float64)
    h1 = 2 * cb / (ca + cc + cd) ** 3
    h2 = -1.0 / (ca + cc + cd) ** 2
    idx = 0
    irow[idx], jcol[idx], data[idx] = (2, 2, lam[0] * h1)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (3, 2, lam[0] * h2)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (4, 2, lam[0] * h1)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (4, 3, lam[0] * h2)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (4, 4, lam[0] * h1)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (5, 2, lam[0] * h1)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (5, 3, lam[0] * h2)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (5, 4, lam[0] * h1)
    idx += 1
    irow[idx], jcol[idx], data[idx] = (5, 5, lam[0] * h1)
    idx += 1
    assert idx == nnz
    hess = coo_matrix((data, (irow, jcol)), shape=(6, 6))
    return hess