from pyomo.common import unittest
from pyomo.contrib.pynumero.dependencies import numpy_available, scipy_available
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from scipy.sparse import coo_matrix, spmatrix
import numpy as np
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.linalg.ma57_interface import MA57
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
from pyomo.contrib.pynumero.linalg.scipy_interface import ScipyLU, ScipyIterative
from scipy.sparse.linalg import gmres
from pyomo.contrib.pynumero.linalg.mumps_interface import (
def singular_helper(self, solver: LinearSolverInterface):
    m = np.array([[1, 1], [1, 1]], dtype=np.double)
    x = np.array([4, 7], dtype=np.double)
    bm, bx, br = self.create_blocks(m, x)
    br.get_block(0)[1] += 1
    br.get_block(1)[1] += 1
    bx2, res = solver.solve(bm, br, raise_on_error=False)
    self.assertNotEqual(res.status, LinearSolverStatus.successful)