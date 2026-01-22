import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def test_indexed_constraint(self):
    m = ConcreteModel()
    m.x = Var([0, 1, 2, 3])
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.array([10, 20])
    m.c = Constraint([0, 1], expr=A @ m.x <= b)
    self.assertTrue(compare_expressions(m.c[0].expr, m.x[0] + 2 * m.x[1] + 3 * m.x[2] + 4 * m.x[3] <= 10))
    self.assertTrue(compare_expressions(m.c[1].expr, 5 * m.x[0] + 6 * m.x[1] + 7 * m.x[2] + 8 * m.x[3] <= 20))