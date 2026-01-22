import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.network import Port, Arc
def test_add_scalar_vars(self):
    pipe = ConcreteModel()
    pipe.flow = Var()
    pipe.pIn = Var(within=NonNegativeReals)
    pipe.pOut = Var(within=NonNegativeReals)
    pipe.OUT = Port()
    pipe.OUT.add(pipe.flow, 'flow')
    pipe.OUT.add(pipe.pOut, 'pressure')
    self.assertEqual(len(pipe.OUT), 1)
    self.assertEqual(len(pipe.OUT.vars), 2)
    self.assertFalse(pipe.OUT.vars['flow'].is_expression_type())
    pipe.IN = Port()
    pipe.IN.add(-pipe.flow, 'flow')
    pipe.IN.add(pipe.pIn, 'pressure')
    self.assertEqual(len(pipe.IN), 1)
    self.assertEqual(len(pipe.IN.vars), 2)
    self.assertTrue(pipe.IN.vars['flow'].is_expression_type())