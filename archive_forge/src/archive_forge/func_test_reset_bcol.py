import warnings
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_reset_bcol(self):
    row = np.array([0, 3, 1, 2, 3, 0])
    col = np.array([0, 0, 1, 2, 3, 3])
    data = np.array([2.0, 1, 3, 4, 5, 1])
    m = coo_matrix((data, (row, col)), shape=(4, 4))
    rank = comm.Get_rank()
    rank_ownership = [[0, -1], [-1, 1]]
    bm = MPIBlockMatrix(2, 2, rank_ownership, comm)
    if rank == 0:
        bm.set_block(0, 0, m)
    if rank == 1:
        bm.set_block(1, 1, m)
    serial_bm = BlockMatrix(2, 2)
    serial_bm.set_block(0, 0, m)
    serial_bm.set_block(1, 1, m)
    self.assertTrue(np.allclose(serial_bm.row_block_sizes(), bm.row_block_sizes()))
    bm.reset_bcol(0)
    serial_bm.reset_bcol(0)
    self.assertTrue(np.allclose(serial_bm.col_block_sizes(), bm.col_block_sizes()))
    bm.reset_bcol(1)
    serial_bm.reset_bcol(1)
    self.assertTrue(np.allclose(serial_bm.col_block_sizes(), bm.col_block_sizes()))