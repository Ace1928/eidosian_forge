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
def check_components_descended_from_model(model, components, components_name, config):
    """
    Check all members in a provided sequence of Pyomo component
    objects are descended from a given ConcreteModel object.

    Parameters
    ----------
    model : ConcreteModel
        Model from which components should all be descended.
    components : Iterable of Component
        Components of interest.
    components_name : str
        Brief description or name for the sequence of components.
        Used for constructing error messages.
    config : ConfigDict
        PyROS solver options.

    Raises
    ------
    ValueError
        If at least one entry of `components` is not descended
        from `model`.
    """
    components_not_in_model = [comp for comp in components if comp.model() is not model]
    if components_not_in_model:
        comp_names_str = '\n '.join((f'{comp.name!r}, from model with name {comp.model().name!r}' for comp in components_not_in_model))
        config.progress_logger.error(f'The following {components_name} are not descended from the input deterministic model with name {model.name!r}:\n {comp_names_str}')
        raise ValueError(f'Found entries of {components_name} not descended from input model. Check logger output messages.')