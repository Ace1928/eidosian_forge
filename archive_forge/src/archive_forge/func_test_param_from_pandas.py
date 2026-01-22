import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
@unittest.skipUnless(pandas_available, 'pandas is not available')
def test_param_from_pandas(self):
    model = ConcreteModel()
    model.I = Set(initialize=range(6))
    model.P0 = Param(model.I, initialize={0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0})
    model.P1 = Param(model.I, initialize=pd.Series({0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0}).to_dict())
    model.P2 = Param(model.I, initialize=pd.Series({0: 400.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 240.0}))
    self.assertEqual(list(model.P0.values()), list(model.P1.values()))
    self.assertEqual(list(model.P0.values()), list(model.P2.values()))
    model.V = Var(model.I, initialize=0)

    def rule(m, l):
        return -m.P0[l] <= m.V[l]
    model.Constraint0 = Constraint(model.I, rule=rule)

    def rule(m, l):
        return -m.P1[l] <= m.V[l]
    model.Constraint1 = Constraint(model.I, rule=rule)

    def rule(m, l):
        return -m.P2[l] <= m.V[l]
    model.Constraint2 = Constraint(model.I, rule=rule)