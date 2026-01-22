import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_set_blocks(self):
    v = self.ones
    blocks = [np.ones(s) * i for i, s in enumerate(self.list_sizes_ones)]
    v.set_blocks(blocks)
    for i, s in enumerate(self.list_sizes_ones):
        self.assertEqual(v.get_block(i).size, s)
        self.assertEqual(v.get_block(i).shape, (s,))
        res = np.ones(s) * i
        self.assertListEqual(v.get_block(i).tolist(), res.tolist())