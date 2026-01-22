import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
@cached_property
def need_shaped_indexing(self):
    """
        Whether iterating on this iterator requires keeping track of
        individual indices inside the shape.  If False, only a single index
        over the equivalent flat shape is required, which can make the
        iterator more efficient.
        """
    for kind, start_dim, end_dim, _ in self.indexers:
        if kind in ('0d', 'scalar'):
            pass
        elif kind == 'flat':
            if (start_dim, end_dim) != (0, self.ndim):
                return True
        else:
            return True
    return False