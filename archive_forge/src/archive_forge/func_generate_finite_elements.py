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
def generate_finite_elements(ds, nfe):
    """
    This function first checks to see if the number of finite elements
    in the differential set is equal to nfe. If the number of finite
    elements is less than nfe, additional points will be generated. If
    the number of finite elements is greater than or equal to nfe the
    differential set will not be modified
    """
    if len(ds) - 1 >= nfe:
        return
    elif len(ds) == 2:
        step = (max(ds) - min(ds)) / float(nfe)
        tmp = min(ds) + step
        while round(tmp, 6) <= round(max(ds) - step, 6):
            ds.add(round(tmp, 6))
            tmp += step
        ds.set_changed(True)
        ds._fe = list(ds)
        return
    else:
        addpts = nfe - (len(ds) - 1)
        while addpts > 0:
            _add_point(ds)
            addpts -= 1
        ds.set_changed(True)
        ds._fe = list(ds)
        return