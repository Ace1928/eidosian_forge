import collections.abc
import pickle
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.block import block, block_dict
def test_setitem_exists_overwrite(self):
    index = ['a', 1, None, (1,), (1, 2)]
    c = self._container_type(((i, self._ctype_factory()) for i in index))
    self.assertEqual(len(c), len(index))
    for i in index:
        self.assertTrue(i in c)
        cdata = c[i]
        c[i] = self._ctype_factory()
        self.assertEqual(len(c), len(index))
        self.assertTrue(i in c)
        self.assertNotEqual(id(cdata), id(c[i]))
        self.assertEqual(cdata.parent, None)