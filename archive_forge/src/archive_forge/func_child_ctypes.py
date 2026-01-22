import logging
import math
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.base import _no_ctype, _convert_ctype
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.container_utils import define_simple_containers
def child_ctypes(self):
    """Returns the set of child object category types
        stored in this container."""
    self_byctype = self.__byctype
    if self_byctype is None:
        return ()
    elif self_byctype.__class__ is int:
        ctypes_set = set()
        ctypes = []
        for child in self.__order.values():
            child_ctype = child.ctype
            if child_ctype not in ctypes_set:
                ctypes_set.add(child_ctype)
                ctypes.append(child_ctype)
        return tuple(ctypes)
    elif self_byctype.__class__ is dict:
        return tuple(self_byctype)
    else:
        return (self_byctype,)