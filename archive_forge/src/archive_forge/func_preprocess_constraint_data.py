import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
def preprocess_constraint_data(block, constraint_data, idMap=None, block_repn=None):
    if not hasattr(block, '_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn
    if constraint_data.body is None:
        raise ValueError('No expression has been defined for the body of constraint %s' % constraint_data.name)
    try:
        repn = generate_standard_repn(constraint_data.body, idMap=idMap)
    except Exception:
        err = sys.exc_info()[1]
        logging.getLogger('pyomo.core').error('exception generating a standard representation for constraint %s: %s' % (constraint_data.name, str(err)))
        raise
    block_repn[constraint_data] = repn