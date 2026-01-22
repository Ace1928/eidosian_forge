import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_initialize_param_from_ndarray(self):
    samples = 10
    c1 = 0.5
    c2 = 0.5
    model = ConcreteModel()
    model.i = RangeSet(samples)

    def init_x(model, i):
        return np.random.rand(1)

    def init_y(model, i):
        return c1 * model.x[i] ** 2 + c2 * model.x[i]
    model.x = Param(model.i, initialize=init_x)
    model.y = Param(model.i, initialize=init_y, domain=Reals)
    model.c_1 = Var(initialize=1)
    model.c_2 = Var(initialize=1)
    model.error = Objective(expr=sum(((model.c_1 * model.x[i] ** 2 + model.c_2 * model.x[i] - model.y[i]) ** 2 for i in model.i)))
    repn = generate_standard_repn(model.error.expr, compute_values=True)
    self.assertIsNone(repn.nonlinear_expr)
    self.assertEqual(len(repn.quadratic_vars), 3)
    for i in range(3):
        self.assertGreater(repn.quadratic_coefs[i], 0)
    self.assertEqual(len(repn.linear_vars), 2)
    for i in range(2):
        self.assertLess(repn.linear_coefs[i], 0)
    self.assertGreater(repn.constant, 0)