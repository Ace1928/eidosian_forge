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
def indexers(self):
    """
        A list of (kind, start_dim, end_dim, indices) where:
        - `kind` is either "flat", "indexed", "0d" or "scalar"
        - `start_dim` and `end_dim` are the dimension numbers at which
          this indexing takes place
        - `indices` is the indices of the indexed arrays in self.arrays
        """
    d = collections.OrderedDict()
    layout = self.layout
    ndim = self.ndim
    assert layout in 'CF'
    for i, a in enumerate(self.arrays):
        if not isinstance(a, Array):
            indexer = ('scalar', 0, 0)
        elif a.ndim == 0:
            indexer = ('0d', 0, 0)
        else:
            if a.layout == layout or (a.ndim == 1 and a.layout in 'CF'):
                kind = 'flat'
            else:
                kind = 'indexed'
            if layout == 'C':
                indexer = (kind, ndim - a.ndim, ndim)
            else:
                indexer = (kind, 0, a.ndim)
        d.setdefault(indexer, []).append(i)
    return list((k + (v,) for k, v in d.items()))