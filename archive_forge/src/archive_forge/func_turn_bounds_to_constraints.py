import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def turn_bounds_to_constraints(variable, model, config=None):
    """
    Turn the variable in question's "bounds" into direct inequality constraints on the model.
    :param variable: the variable with bounds to be turned to None and made into constraints.
    :param model: the model in which the variable resides
    :param config: solver config
    :return: the list of inequality constraints that are the bounds
    """
    lb, ub = (variable.lower, variable.upper)
    if variable.domain is not Reals:
        variable.domain = Reals
    if isinstance(lb, NPV_MaxExpression):
        lb_args = lb.args
    else:
        lb_args = (lb,)
    if isinstance(ub, NPV_MinExpression):
        ub_args = ub.args
    else:
        ub_args = (ub,)
    count = 0
    for arg in lb_args:
        if arg is not None:
            name = unique_component_name(model, variable.name + f'_lower_bound_con_{count}')
            model.add_component(name, Constraint(expr=arg - variable <= 0))
            count += 1
            variable.setlb(None)
    count = 0
    for arg in ub_args:
        if arg is not None:
            name = unique_component_name(model, variable.name + f'_upper_bound_con_{count}')
            model.add_component(name, Constraint(expr=variable - arg <= 0))
            count += 1
            variable.setub(None)