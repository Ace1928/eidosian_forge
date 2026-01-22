import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.sparse.block_vector import (
def test_newbyteorder(self):
    v = self.ones
    with self.assertRaises(NotImplementedError) as ctx:
        vv = v.newbyteorder()