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
def block_fully_discretized(b):
    """
    Checks to see if all ContinuousSets in a block have been discretized
    """
    for i in b.component_map(ContinuousSet).values():
        if 'scheme' not in i.get_discretization_info():
            return False
    return True