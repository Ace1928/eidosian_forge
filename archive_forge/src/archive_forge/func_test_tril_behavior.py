import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
import numpy as np
from scipy.sparse import coo_matrix, tril
from pyomo.contrib import interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
def test_tril_behavior(self):
    mat = get_base_matrix(use_tril=True)
    mat2 = tril(mat)
    self.assertTrue(np.all(mat.row == mat2.row))
    self.assertTrue(np.all(mat.col == mat2.col))
    self.assertTrue(np.allclose(mat.data, mat2.data))
    mat = get_base_matrix_wrong_order(use_tril=True)
    self.assertFalse(np.all(mat.row == mat2.row))
    self.assertFalse(np.allclose(mat.data, mat2.data))
    mat2 = tril(mat)
    self.assertTrue(np.all(mat.row == mat2.row))
    self.assertTrue(np.all(mat.col == mat2.col))
    self.assertTrue(np.allclose(mat.data, mat2.data))