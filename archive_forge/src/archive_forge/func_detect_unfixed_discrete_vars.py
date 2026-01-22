from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException, DeveloperError
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Constraint, TransformationFactory, Objective, Block
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc
def detect_unfixed_discrete_vars(model):
    """Detect unfixed discrete variables in use on the model."""
    var_set = ComponentSet()
    for constr in model.component_data_objects(Constraint, active=True, descend_into=True):
        var_set.update((v for v in EXPR.identify_variables(constr.body, include_fixed=False) if not v.is_continuous()))
    for obj in model.component_data_objects(Objective, active=True):
        var_set.update((v for v in EXPR.identify_variables(obj.expr, include_fixed=False) if not v.is_continuous()))
    return var_set