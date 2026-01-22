import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def update_contset_indexed_component(comp, expansion_map):
    """
    Update any model components which are indexed by a ContinuousSet that
    has changed
    """
    if comp.ctype is Suffix:
        return
    if comp.ctype is Param:
        return
    from pyomo.dae import Integral
    if comp.ctype is Integral:
        return
    if not hasattr(comp, 'dim'):
        return
    if comp.dim() == 0:
        return
    temp = comp.index_set()
    indexset = list(comp.index_set().subsets())
    for s in indexset:
        if s.ctype == ContinuousSet and s.get_changed():
            if isinstance(comp, Var):
                expansion_map[comp] = _update_var
                _update_var(comp)
            elif comp.ctype == Constraint:
                expansion_map[comp] = _update_constraint
                _update_constraint(comp)
            elif comp.ctype == Expression:
                expansion_map[comp] = _update_expression
                _update_expression(comp)
            elif isinstance(comp, Piecewise):
                expansion_map[comp] = _update_piecewise
                _update_piecewise(comp)
            elif comp.ctype == Block:
                expansion_map[comp] = _update_block
                _update_block(comp)
            else:
                raise TypeError('Found component %s of type %s indexed by a ContinuousSet. Components of this type are not currently supported by the automatic discretization transformation in pyomo.dae. Try adding the component to the model after discretizing. Alert the pyomo developers for more assistance.' % (str(comp), comp.ctype))