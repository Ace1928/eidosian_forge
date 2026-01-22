from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
@deprecated('log_active_constraints is deprecated.  Please use pyomo.util.blockutil.log_model_constraints()', version='5.7.3')
def log_active_constraints(m, logger=logger):
    log_model_constraints(m, logger)