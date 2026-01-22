import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet
def identify_inconsistent_units(block):
    """
    This function generates a ComponentSet of all Constraints, Expressions, and Objectives
    in a Block or model which have inconsistent units.

    Parameters
    ----------
    block : Pyomo Block or Model to test

    Returns
    ------
    ComponentSet : contains all Constraints, Expressions or Objectives which were
        identified as having unit consistency issues
    """
    inconsistent_units = ComponentSet()
    for obj in block.component_data_objects([Constraint, Expression, Objective], descend_into=True):
        try:
            assert_units_consistent(obj)
        except UnitsError:
            inconsistent_units.add(obj)
    return inconsistent_units