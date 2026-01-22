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
def assert_units_consistent(obj):
    """
    This method raises an exception if the units are not
    consistent on the passed in object.  Argument obj can be one
    of the following components: Pyomo Block (or Model),
    Constraint, Objective, Expression, or it can be a Pyomo
    expression object

    Parameters
    ----------
    obj : Pyomo component (e.g., Block, Model, Constraint, Objective, or Expression) or Pyomo expression
       The object or expression to test

    Raises
    ------
    :py:class:`pyomo.core.base.units_container.UnitsError`, :py:class:`pyomo.core.base.units_container.InconsistentUnitsError`
    """
    objtype = type(obj)
    if objtype in native_types:
        return
    elif obj.is_expression_type() or objtype is IndexTemplate:
        try:
            _assert_units_consistent_expression(obj)
        except UnitsError:
            logger.error('Units problem with expression {}'.format(obj))
            raise
        return
    if obj.ctype not in _component_data_handlers:
        raise TypeError('Units checking not supported for object of type {}.'.format(obj.ctype))
    handler = _component_data_handlers[obj.ctype]
    if handler is None:
        return
    if obj.is_indexed():
        for cdata in obj.values():
            try:
                handler(cdata)
            except UnitsError:
                logger.error('Error in units when checking {}'.format(cdata))
                raise
    else:
        handler(obj)