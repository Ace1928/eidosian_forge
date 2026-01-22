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
def get_all_specialized_permutations(self, fused_types=None):
    """
        Permute all the types. For every specific instance of a fused type, we
        want all other specific instances of all other fused types.

        It returns an iterable of two-tuples of the cname that should prefix
        the cname of the function, and a dict mapping any fused types to their
        respective specific types.
        """
    assert self.is_fused
    if fused_types is None:
        fused_types = self.get_fused_types()
    return get_all_specialized_permutations(fused_types)