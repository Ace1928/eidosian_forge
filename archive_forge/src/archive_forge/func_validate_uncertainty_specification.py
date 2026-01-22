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
def validate_uncertainty_specification(model, config):
    """
    Validate specification of uncertain parameters and uncertainty
    set.

    Parameters
    ----------
    model : ConcreteModel
        Input deterministic model.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one of the following holds:

        - dimension of uncertainty set does not equal number of
          uncertain parameters
        - uncertainty set `is_valid()` method does not return
          true.
        - nominal parameter realization is not in the uncertainty set.
    """
    check_components_descended_from_model(model=model, components=config.uncertain_params, components_name='uncertain parameters', config=config)
    if len(config.uncertain_params) != config.uncertainty_set.dim:
        raise ValueError(f'Length of argument `uncertain_params` does not match dimension of argument `uncertainty_set` ({len(config.uncertain_params)} != {config.uncertainty_set.dim}).')
    if not config.uncertainty_set.is_valid(config=config):
        raise ValueError(f'Uncertainty set {config.uncertainty_set} is invalid, as it is either empty or unbounded.')
    if not config.nominal_uncertain_param_vals:
        config.nominal_uncertain_param_vals = [value(param, exception=True) for param in config.uncertain_params]
    elif len(config.nominal_uncertain_param_vals) != len(config.uncertain_params):
        raise ValueError(f'Lengths of arguments `uncertain_params` and `nominal_uncertain_param_vals` do not match ({len(config.uncertain_params)} != {len(config.nominal_uncertain_param_vals)}).')
    nominal_point_in_set = config.uncertainty_set.point_in_set(point=config.nominal_uncertain_param_vals)
    if not nominal_point_in_set:
        raise ValueError(f'Nominal uncertain parameter realization {config.nominal_uncertain_param_vals} is not a point in the uncertainty set {config.uncertainty_set!r}.')