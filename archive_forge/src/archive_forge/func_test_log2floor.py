import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
def test_log2floor(self):
    self.assertEqual(util.log2floor(1), 0)
    self.assertEqual(util.log2floor(2), 1)
    self.assertEqual(util.log2floor(3), 1)
    self.assertEqual(util.log2floor(4), 2)
    self.assertEqual(util.log2floor(5), 2)
    self.assertEqual(util.log2floor(6), 2)
    self.assertEqual(util.log2floor(7), 2)
    self.assertEqual(util.log2floor(8), 3)
    self.assertEqual(util.log2floor(9), 3)
    self.assertEqual(util.log2floor(2 ** 10), 10)
    self.assertEqual(util.log2floor(2 ** 10 + 1), 10)
    self.assertEqual(util.log2floor(2 ** 20), 20)
    self.assertEqual(util.log2floor(2 ** 20 + 1), 20)
    self.assertEqual(util.log2floor(2 ** 30), 30)
    self.assertEqual(util.log2floor(2 ** 30 + 1), 30)
    self.assertEqual(util.log2floor(2 ** 40), 40)
    self.assertEqual(util.log2floor(2 ** 40 + 1), 40)