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
def test_clone_unclonable_attribute(self):

    class foo(object):

        def __deepcopy__(bogus):
            pass
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1])
    m.bad1 = foo()
    m.c = Constraint(expr=m.x ** 2 + m.y[1] <= 5)
    m.b = Block()
    m.b.x = Var()
    m.b.y = Var([1, 2])
    m.b.bad2 = foo()
    m.b.c = Constraint(expr=m.x ** 2 + m.y[1] + m.b.x ** 2 + m.b.y[1] <= 10)
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        nb = deepcopy(m.b)
    self.assertIn("'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue())
    self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
    self.assertIn("'__paranoid__'", OUTPUT.getvalue())
    self.assertTrue(hasattr(m.b, 'bad2'))
    self.assertIsNotNone(m.b.bad2)
    self.assertTrue(hasattr(nb, 'bad2'))
    self.assertIsNone(nb.bad2)
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        nb = m.b.clone()
    self.assertNotIn("'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue())
    self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
    self.assertNotIn("'__paranoid__'", OUTPUT.getvalue())
    self.assertTrue(hasattr(m.b, 'bad2'))
    self.assertIsNotNone(m.b.bad2)
    self.assertTrue(hasattr(nb, 'bad2'))
    self.assertIsNone(nb.bad2)
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.core'):
        n = m.clone()
    self.assertIn("'unknown' contains an uncopyable field 'bad1'", OUTPUT.getvalue())
    self.assertIn("'b' contains an uncopyable field 'bad2'", OUTPUT.getvalue())
    self.assertNotIn("'__paranoid__'", OUTPUT.getvalue())
    self.assertTrue(hasattr(m, 'bad1'))
    self.assertIsNotNone(m.bad1)
    self.assertTrue(hasattr(n, 'bad1'))
    self.assertIsNone(n.bad1)
    self.assertTrue(hasattr(m.b, 'bad2'))
    self.assertIsNotNone(m.b.bad2)
    self.assertTrue(hasattr(n.b, 'bad2'))
    self.assertIsNone(n.b.bad2)
    self.assertNotEqual(id(m), id(n))
    self.assertNotEqual(id(m.x), id(n.x))
    self.assertIs(m.x.parent_block(), m)
    self.assertIs(m.x.parent_component(), m.x)
    self.assertIs(n.x.parent_block(), n)
    self.assertIs(n.x.parent_component(), n.x)
    self.assertNotEqual(id(m.y), id(n.y))
    self.assertIs(m.y.parent_block(), m)
    self.assertIs(m.y[1].parent_component(), m.y)
    self.assertIs(n.y.parent_block(), n)
    self.assertIs(n.y[1].parent_component(), n.y)
    self.assertNotEqual(id(m.c), id(n.c))
    self.assertIs(m.c.parent_block(), m)
    self.assertIs(m.c.parent_component(), m.c)
    self.assertIs(n.c.parent_block(), n)
    self.assertIs(n.c.parent_component(), n.c)
    self.assertEqual(sorted((id(x) for x in EXPR.identify_variables(m.c.body))), sorted((id(x) for x in (m.x, m.y[1]))))
    self.assertEqual(sorted((id(x) for x in EXPR.identify_variables(n.c.body))), sorted((id(x) for x in (n.x, n.y[1]))))
    self.assertNotEqual(id(m.b), id(n.b))
    self.assertIs(m.b.parent_block(), m)
    self.assertIs(m.b.parent_component(), m.b)
    self.assertIs(n.b.parent_block(), n)
    self.assertIs(n.b.parent_component(), n.b)
    self.assertNotEqual(id(m.b.x), id(n.b.x))
    self.assertIs(m.b.x.parent_block(), m.b)
    self.assertIs(m.b.x.parent_component(), m.b.x)
    self.assertIs(n.b.x.parent_block(), n.b)
    self.assertIs(n.b.x.parent_component(), n.b.x)
    self.assertNotEqual(id(m.b.y), id(n.b.y))
    self.assertIs(m.b.y.parent_block(), m.b)
    self.assertIs(m.b.y[1].parent_component(), m.b.y)
    self.assertIs(n.b.y.parent_block(), n.b)
    self.assertIs(n.b.y[1].parent_component(), n.b.y)
    self.assertNotEqual(id(m.b.c), id(n.b.c))
    self.assertIs(m.b.c.parent_block(), m.b)
    self.assertIs(m.b.c.parent_component(), m.b.c)
    self.assertIs(n.b.c.parent_block(), n.b)
    self.assertIs(n.b.c.parent_component(), n.b.c)
    self.assertEqual(sorted((id(x) for x in EXPR.identify_variables(m.b.c.body))), sorted((id(x) for x in (m.x, m.y[1], m.b.x, m.b.y[1]))))
    self.assertEqual(sorted((id(x) for x in EXPR.identify_variables(n.b.c.body))), sorted((id(x) for x in (n.x, n.y[1], n.b.x, n.b.y[1]))))