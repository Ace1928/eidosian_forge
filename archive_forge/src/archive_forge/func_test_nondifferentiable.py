import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_nondifferentiable(self):
    m = ConcreteModel()
    m.foo = Var()
    self.assertRaisesRegex(NondifferentiableError, "The sub-expression '.*' is not differentiable with respect to .*foo", differentiate, ceil(m.foo), wrt=m.foo)
    self.assertRaisesRegex(NondifferentiableError, "The sub-expression '.*' is not differentiable with respect to .*foo", differentiate, floor(m.foo), wrt=m.foo)