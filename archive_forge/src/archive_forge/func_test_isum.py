import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_isum(self):
    v = MPIBlockVector(3, [0, 1, -1], comm)
    rank = comm.Get_rank()
    if rank == 0:
        v.set_block(0, np.arange(3))
    if rank == 1:
        v.set_block(1, np.arange(4))
    v.set_block(2, np.arange(2))
    v += v
    self.assertTrue(isinstance(v, MPIBlockVector))
    self.assertEqual(3, v.nblocks)
    if rank == 0:
        self.assertTrue(np.allclose(np.arange(3) * 2.0, v.get_block(0)))
    if rank == 1:
        self.assertTrue(np.allclose(np.arange(4) * 2.0, v.get_block(1)))
    self.assertTrue(np.allclose(np.arange(2) * 2.0, v.get_block(2)))
    v = MPIBlockVector(3, [0, 1, -1], comm)
    rank = comm.Get_rank()
    if rank == 0:
        v.set_block(0, np.arange(3))
    if rank == 1:
        v.set_block(1, np.arange(4))
    v.set_block(2, np.arange(2))
    v = MPIBlockVector(3, [0, 1, -1], comm)
    rank = comm.Get_rank()
    if rank == 0:
        v.set_block(0, np.arange(3, dtype='d'))
    if rank == 1:
        v.set_block(1, np.arange(4, dtype='d'))
    v.set_block(2, np.arange(2, dtype='d'))
    v += 7.0
    self.assertTrue(isinstance(v, MPIBlockVector))
    self.assertEqual(3, v.nblocks)
    if rank == 0:
        self.assertTrue(np.allclose(np.arange(3) + 7.0, v.get_block(0)))
    if rank == 1:
        self.assertTrue(np.allclose(np.arange(4) + 7.0, v.get_block(1)))
    self.assertTrue(np.allclose(np.arange(2) + 7.0, v.get_block(2)))