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
@unittest.skipUnless(util.numpy_available and util.scipy_available, 'Numpy or Scipy is not available')
def test_generate_delaunay(self):
    vlist = variable_list()
    vlist.append(variable(lb=0, ub=1))
    vlist.append(variable(lb=1, ub=2))
    vlist.append(variable(lb=2, ub=3))
    if not (util.numpy_available and util.scipy_available):
        with self.assertRaises(ImportError):
            util.generate_delaunay(vlist)
    else:
        tri = util.generate_delaunay(vlist, num=2)
        self.assertTrue(isinstance(tri, util.scipy.spatial.Delaunay))
        self.assertEqual(len(tri.simplices), 6)
        self.assertEqual(len(tri.points), 8)
        tri = util.generate_delaunay(vlist, num=3)
        self.assertTrue(isinstance(tri, util.scipy.spatial.Delaunay))
        self.assertEqual(len(tri.simplices), 62)
        self.assertEqual(len(tri.points), 27)
    vlist = variable_list()
    vlist.append(variable(lb=0))
    with self.assertRaises(ValueError):
        util.generate_delaunay(vlist)
    vlist = variable_list()
    vlist.append(variable(ub=0))
    with self.assertRaises(ValueError):
        util.generate_delaunay(vlist)