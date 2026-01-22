import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_decomposable_matrix_permuted(self):
    """
        Same matrix as above, but now permuted into a random order.
        """
    row = [0, 1, 1, 2, 2, 3, 3, 4]
    col = [0, 0, 1, 2, 3, 3, 4, 4]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    N = 5
    row_perm = list(range(N))
    col_perm = list(range(N))
    random.shuffle(row_perm)
    random.shuffle(col_perm)
    row = [row_perm[i] for i in row]
    col = [col_perm[i] for i in col]
    coo = sp.sparse.coo_matrix((data, (row, col)), shape=(N, N))
    row_blocks, col_blocks = get_independent_submatrices(coo)
    self.assertEqual(len(row_blocks), 2)
    self.assertEqual(len(col_blocks), 2)
    row_set_1 = set((row_perm[0], row_perm[1]))
    row_set_2 = set((row_perm[2], row_perm[3], row_perm[4]))
    col_set_1 = set((col_perm[0], col_perm[1]))
    col_set_2 = set((col_perm[2], col_perm[3], col_perm[4]))
    self.assertTrue(set(row_blocks[0]) == row_set_1 or set(row_blocks[1]) == row_set_1)
    self.assertTrue(set(col_blocks[0]) == col_set_1 or set(col_blocks[1]) == col_set_1)
    self.assertTrue(set(row_blocks[0]) == row_set_2 or set(row_blocks[1]) == row_set_2)
    self.assertTrue(set(col_blocks[0]) == col_set_2 or set(col_blocks[1]) == col_set_2)