from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def get_all_specialized_function_types(self):
    """
        Get all the specific function types of this one.
        """
    assert self.is_fused
    if self.entry.fused_cfunction:
        return [n.type for n in self.entry.fused_cfunction.nodes]
    elif self.cached_specialized_types is not None:
        return self.cached_specialized_types
    result = []
    permutations = self.get_all_specialized_permutations()
    new_cfunc_entries = []
    for cname, fused_to_specific in permutations:
        new_func_type = self.entry.type.specialize(fused_to_specific)
        if self.optional_arg_count:
            self.declare_opt_arg_struct(new_func_type, cname)
        new_entry = copy.deepcopy(self.entry)
        new_func_type.specialize_entry(new_entry, cname)
        new_entry.type = new_func_type
        new_func_type.entry = new_entry
        result.append(new_func_type)
        new_cfunc_entries.append(new_entry)
    cfunc_entries = self.entry.scope.cfunc_entries
    try:
        cindex = cfunc_entries.index(self.entry)
    except ValueError:
        cfunc_entries.extend(new_cfunc_entries)
    else:
        cfunc_entries[cindex:cindex + 1] = new_cfunc_entries
    self.cached_specialized_types = result
    return result