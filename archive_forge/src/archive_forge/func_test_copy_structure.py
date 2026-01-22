import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_copy_structure(self):
    v = BlockVector(2)
    a = np.ones(5)
    b = np.zeros(9)
    v.set_block(0, a)
    v.set_block(1, b)
    v2 = v.copy_structure()
    self.assertEqual(v.get_block(0).size, v2.get_block(0).size)
    self.assertEqual(v.get_block(1).size, v2.get_block(1).size)