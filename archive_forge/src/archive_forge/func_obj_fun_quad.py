import pyomo.common.unittest as unittest
from pyomo.common.dependencies import (
from pyomo.environ import (
from pyomo.core.expr import MonomialTermExpression
from pyomo.core.expr.ndarray import NumericNDArray
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.expr.compare import compare_expressions
from pyomo.core.expr.relational_expr import InequalityExpression
from pyomo.repn import generate_standard_repn
def obj_fun_quad(model):
    return sum(((model.Y[i - 1] - (model.beta0 + sum((model.X[i - 1, j - 1] * model.beta[j] for j in model.J)))) ** 2 for i in model.I))