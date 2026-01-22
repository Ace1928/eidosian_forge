import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
import pyomo.contrib.pynumero.interfaces.utils as utils
def test_repeated_row_col(self):
    data = [1.0, 0.0, 2.0]
    row = [1, 2, 1]
    col = [2, 2, 2]
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))
    data = [3.0, 0.0]
    row = [1, 2]
    col = [2, 2]
    B = scipy.sparse.coo_matrix((data, (row, col)), shape=(3, 3))
    sparse_sum = utils.CondensedSparseSummation([A, B])
    C = sparse_sum.sum([A, B])
    expected_data = np.asarray([6.0, 0.0], dtype=np.float64)
    expected_row = np.asarray([1, 2], dtype=np.int64)
    expected_col = np.asarray([2, 2], dtype=np.int64)
    self.assertTrue(np.array_equal(expected_data, C.data))
    self.assertTrue(np.array_equal(expected_row, C.row))
    self.assertTrue(np.array_equal(expected_col, C.col))