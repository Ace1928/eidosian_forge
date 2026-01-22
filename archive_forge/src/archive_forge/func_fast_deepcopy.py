import types
from collections import namedtuple
from copy import deepcopy
from weakref import ref as _weakref_ref
def fast_deepcopy(obj, memo):
    """A faster implementation of copy.deepcopy()

    Python's default implementation of deepcopy has several features that
    are slower than they need to be.  This is an implementation of
    deepcopy that provides special handling to circumvent some of the
    slowest parts of deepcopy().

    """
    if obj.__class__ in _atomic_types:
        return obj
    _id = id(obj)
    if _id in memo:
        return memo[_id]
    else:
        return _deepcopy_mapper.get(obj.__class__, _deepcopier)(obj, memo, _id)