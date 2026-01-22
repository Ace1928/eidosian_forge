from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
def test_transfer_attributes_from(self):
    b = Block(concrete=True)
    b.x = Var()
    b.y = Var()
    c = Block(concrete=True)
    c.z = Param(initialize=5)
    c.x = c_x = Param(initialize=5)
    c.y = c_y = 5
    b.clear()
    b.transfer_attributes_from(c)
    self.assertEqual(list(b.component_map()), ['z', 'x'])
    self.assertEqual(list(c.component_map()), [])
    self.assertIs(b.x, c_x)
    self.assertIs(b.y, c_y)

    class DerivedBlock(ScalarBlock):
        _Block_reserved_words = set()

        def __init__(self, *args, **kwds):
            super(DerivedBlock, self).__init__(*args, **kwds)
            with self._declare_reserved_components():
                self.x = Var()
                self.y = Var()
    DerivedBlock._Block_reserved_words = set(dir(DerivedBlock()))
    b = DerivedBlock(concrete=True)
    b_x = b.x
    b_y = b.y
    c = Block(concrete=True)
    c.z = Param(initialize=5)
    c.x = c_x = Param(initialize=5)
    c.y = c_y = 5
    b.clear()
    b.transfer_attributes_from(c)
    self.assertEqual(list(b.component_map()), ['y', 'z', 'x'])
    self.assertEqual(list(c.component_map()), [])
    self.assertIs(b.x, c_x)
    self.assertIsNot(b.y, c_y)
    self.assertIs(b.y, b_y)
    self.assertEqual(value(b.y), value(c_y))
    b = DerivedBlock(concrete=True)
    b_x = b.x
    b_y = b.y
    c = {'z': Param(initialize=5), 'x': Param(initialize=5), 'y': 5}
    b.clear()
    b.transfer_attributes_from(c)
    self.assertEqual(list(b.component_map()), ['y', 'z', 'x'])
    self.assertEqual(sorted(list(c.keys())), ['x', 'y', 'z'])
    self.assertIs(b.x, c['x'])
    self.assertIsNot(b.y, c['y'])
    self.assertIs(b.y, b_y)
    self.assertEqual(value(b.y), value(c_y))
    b = Block(concrete=True)
    b.x = b_x = Var()
    b.y = b_y = Var()
    b.transfer_attributes_from(b)
    self.assertEqual(list(b.component_map()), ['x', 'y'])
    self.assertIs(b.x, b_x)
    self.assertIs(b.y, b_y)
    b = Block(concrete=True)
    b.c = Block()
    b.c.d = Block()
    b.c.d.e = Block()
    with self.assertRaisesRegex(ValueError, '_BlockData.transfer_attributes_from\\(\\): Cannot set a sub-block \\(c.d.e\\) to a parent block \\(c\\):'):
        b.c.d.e.transfer_attributes_from(b.c)
    b = Block(concrete=True)
    with self.assertRaisesRegex(ValueError, '_BlockData.transfer_attributes_from\\(\\): expected a Block or dict; received str'):
        b.transfer_attributes_from('foo')