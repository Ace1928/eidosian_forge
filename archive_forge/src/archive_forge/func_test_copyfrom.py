import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_copyfrom(self):
    v = self.ones
    v1 = np.zeros(v.size)
    v.copyfrom(v1)
    self.assertListEqual(v.tolist(), v1.tolist())
    v2 = BlockVector(len(self.list_sizes_ones))
    for i, s in enumerate(self.list_sizes_ones):
        v2.set_block(i, np.ones(s) * i)
    v.copyfrom(v2)
    for idx, blk in enumerate(v2):
        self.assertListEqual(blk.tolist(), v2.get_block(idx).tolist())
    v3 = BlockVector(2)
    v4 = v.clone(2)
    v3.set_block(0, v4)
    v3.set_block(1, np.zeros(3))
    self.assertListEqual(v3.tolist(), v4.tolist() + [0] * 3)