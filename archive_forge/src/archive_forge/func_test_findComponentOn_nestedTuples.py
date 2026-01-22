import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_findComponentOn_nestedTuples(self):
    m = ConcreteModel()
    m.x = Var()
    m.c = Constraint(Any)
    m.c[0] = m.x >= 0
    m.c[1,] = m.x >= 1
    m.c[2,] = m.x >= 2
    m.c[2] = m.x >= 3
    self.assertIs(ComponentUID(m.c[0]).find_component_on(m), m.c[0])
    self.assertIs(ComponentUID('c[0]').find_component_on(m), m.c[0])
    self.assertIsNone(ComponentUID('c[(0,)]').find_component_on(m))
    self.assertIs(ComponentUID(m.c[1,]).find_component_on(m), m.c[1,])
    self.assertIs(ComponentUID('c[(1,)]').find_component_on(m), m.c[1,])
    self.assertIsNone(ComponentUID('c[1]').find_component_on(m))
    self.assertIs(ComponentUID('c[(2,)]').find_component_on(m), m.c[2,])
    self.assertIs(ComponentUID('c[2]').find_component_on(m), m.c[2])
    self.assertEqual(len(m.c), 4)
    self.assertEqual(repr(ComponentUID(m.c[0])), 'c[0]')
    self.assertEqual(repr(ComponentUID(m.c[1,])), 'c[(1,)]')
    self.assertEqual(str(ComponentUID(m.c[0])), 'c[0]')
    self.assertEqual(str(ComponentUID(m.c[1,])), 'c[(1,)]')
    m = ConcreteModel()
    m.x = Var()
    m.c = Constraint([0, 1])
    m.c[0] = m.x >= 0
    m.c[1,] = m.x >= 1
    self.assertIs(ComponentUID(m.c[0]).find_component_on(m), m.c[0])
    self.assertIs(ComponentUID(m.c[0,]).find_component_on(m), m.c[0])
    self.assertIs(ComponentUID('c[0]').find_component_on(m), m.c[0])
    self.assertIs(ComponentUID('c[(0,)]').find_component_on(m), m.c[0])
    self.assertIs(ComponentUID(m.c[1]).find_component_on(m), m.c[1])
    self.assertIs(ComponentUID(m.c[1,]).find_component_on(m), m.c[1])
    self.assertIs(ComponentUID('c[(1,)]').find_component_on(m), m.c[1])
    self.assertIs(ComponentUID('c[1]').find_component_on(m), m.c[1])
    self.assertEqual(len(m.c), 2)
    m = ConcreteModel()
    m.b = Block(Any)
    m.b[0].c = Block(Any)
    m.b[0].c[0].x = Var()
    m.b[1,].c = Block(Any)
    m.b[1,].c[1,].x = Var()
    ref = m.b[0].c[0].x
    self.assertIs(ComponentUID(ref).find_component_on(m), ref)
    ref = 'm.b[0].c[(0,)].x'
    self.assertIsNone(ComponentUID(ref).find_component_on(m))
    ref = m.b[1,].c[1,].x
    self.assertIs(ComponentUID(ref).find_component_on(m), ref)
    ref = 'm.b[(1,)].c[1].x'
    self.assertIsNone(ComponentUID(ref).find_component_on(m))
    buf = {}
    ref = m.b[0].c[0].x
    self.assertIs(ComponentUID(ref, cuid_buffer=buf).find_component_on(m), ref)
    self.assertEqual(len(buf), 3)
    ref = 'm.b[0].c[(0,)].x'
    self.assertIsNone(ComponentUID(ref, cuid_buffer=buf).find_component_on(m))
    self.assertEqual(len(buf), 3)
    ref = m.b[1,].c[1,].x
    self.assertIs(ComponentUID(ref, cuid_buffer=buf).find_component_on(m), ref)
    self.assertEqual(len(buf), 4)
    ref = 'm.b[(1,)].c[1].x'
    self.assertIsNone(ComponentUID(ref, cuid_buffer=buf).find_component_on(m))
    self.assertEqual(len(buf), 4)