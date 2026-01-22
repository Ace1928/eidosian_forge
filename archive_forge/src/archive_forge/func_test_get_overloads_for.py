import pyomo.common.unittest as unittest
import inspect
from pyomo.common.pyomo_typing import get_overloads_for
from pyomo.environ import Block
def test_get_overloads_for(self):
    func_list = get_overloads_for(Block.__init__)
    self.assertEqual(len(func_list), 1)
    kwds = inspect.getfullargspec(func_list[0]).kwonlyargs
    self.assertEqual(kwds, ['rule', 'concrete', 'dense', 'name', 'doc'])