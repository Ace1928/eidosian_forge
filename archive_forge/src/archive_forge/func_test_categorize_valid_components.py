import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_categorize_valid_components(self):
    m = ConcreteModel()
    m.x = Var()
    m.o = Objective()
    m.b2 = Block()
    m.b2.e = Expression()
    m.b2.p = Param()
    m.b2.q = Param()
    m.b = Block()
    m.b.p = Param()
    m.s = Suffix()
    m.b.t = Suffix()
    m.b.s = Suffix()
    m.b.deactivate()
    component_map, unrecognized = categorize_valid_components(m, valid={Var, Block}, targets={Param, Objective, Set})
    self.assertStructuredAlmostEqual(component_map, {Param: [m.b2], Objective: [m], Set: []})
    self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s]})
    component_map, unrecognized = categorize_valid_components(m, active=None, valid={Var, Block}, targets={Param, Objective, Set})
    self.assertStructuredAlmostEqual(component_map, {Param: [m.b2, m.b], Objective: [m], Set: []})
    self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s, m.b.t, m.b.s], Expression: [m.b2.e]})
    component_map, unrecognized = categorize_valid_components(m, sort=True, valid={Var, Block}, targets={Param, Objective, Set})
    self.assertStructuredAlmostEqual(component_map, {Param: [m.b2], Objective: [m], Set: []})
    self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s]})
    component_map, unrecognized = categorize_valid_components(m, sort=True, active=None, valid={Var, Block}, targets={Param, Objective, Set})
    self.assertStructuredAlmostEqual(component_map, {Param: [m.b, m.b2], Objective: [m], Set: []})
    self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s, m.b.s, m.b.t], Expression: [m.b2.e]})
    with self.assertRaises(AssertionError):
        component_map, unrecognized = categorize_valid_components(m, active=False)
    with self.assertRaisesRegex(DeveloperError, "categorize_valid_components: Cannot have component type \\[\\<class[^>]*Set'\\>\\] in both the `valid` and `targets` sets"):
        categorize_valid_components(m, valid={Var, Block, Set}, targets={Param, Objective, Set})