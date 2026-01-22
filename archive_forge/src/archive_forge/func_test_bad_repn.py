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
def test_bad_repn(self):
    repn = list(transforms_nd.registered_transforms.keys())[0]
    self.assertTrue(repn in transforms_nd.registered_transforms)
    transforms_nd.piecewise_nd(_test_tri, _test_values, repn=repn)
    repn = '_bad_repn_'
    self.assertFalse(repn in transforms_nd.registered_transforms)
    with self.assertRaises(ValueError):
        transforms_nd.piecewise_nd(_test_tri, _test_values, repn=repn)