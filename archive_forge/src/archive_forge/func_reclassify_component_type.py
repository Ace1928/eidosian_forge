import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
def reclassify_component_type(self, name_or_object, new_ctype, preserve_declaration_order=True):
    """
        TODO
        """
    obj = self.component(name_or_object)
    if obj is None:
        return
    if obj.ctype is new_ctype:
        return
    name = obj.local_name
    if not preserve_declaration_order:
        self.del_component(name)
        obj._ctype = new_ctype
        self.add_component(name, obj)
        return
    idx = self._decl[name]
    ctype_info = self._ctypes[obj.ctype]
    ctype_info[2] -= 1
    if ctype_info[2] == 0:
        del self._ctypes[obj.ctype]
    elif ctype_info[0] == idx:
        ctype_info[0] = self._decl_order[idx][1]
    else:
        prev = None
        tmp = self._ctypes[obj.ctype][0]
        while tmp < idx:
            prev = tmp
            tmp = self._decl_order[tmp][1]
        self._decl_order[prev] = (self._decl_order[prev][0], self._decl_order[idx][1])
        if ctype_info[1] == idx:
            ctype_info[1] = prev
    obj._ctype = new_ctype
    if new_ctype not in self._ctypes:
        self._ctypes[new_ctype] = [idx, idx, 1]
        self._decl_order[idx] = (obj, None)
    elif idx < self._ctypes[new_ctype][0]:
        self._decl_order[idx] = (obj, self._ctypes[new_ctype][0])
        self._ctypes[new_ctype][0] = idx
        self._ctypes[new_ctype][2] += 1
    elif idx > self._ctypes[new_ctype][1]:
        prev = self._ctypes[new_ctype][1]
        self._decl_order[prev] = (self._decl_order[prev][0], idx)
        self._decl_order[idx] = (obj, None)
        self._ctypes[new_ctype][1] = idx
        self._ctypes[new_ctype][2] += 1
    else:
        self._ctypes[new_ctype][2] += 1
        prev = None
        tmp = self._ctypes[new_ctype][0]
        while tmp < idx:
            prev = tmp
            tmp = self._decl_order[tmp][1]
        self._decl_order[prev] = (self._decl_order[prev][0], idx)
        self._decl_order[idx] = (obj, tmp)