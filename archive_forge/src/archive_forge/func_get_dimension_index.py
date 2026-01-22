import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
def get_dimension_index(self, dimension):
    """Get the index of the requested dimension.

        Args:
            dimension: Dimension to look up by name or by index

        Returns:
            Integer index of the requested dimension
        """
    if isinstance(dimension, int):
        if dimension < self.ndims + len(self.vdims) or dimension < len(self.dimensions()):
            return dimension
        else:
            return IndexError('Dimension index out of bounds')
    dim = dimension_name(dimension)
    try:
        dimensions = self.kdims + self.vdims
        return next((i for i, d in enumerate(dimensions) if d == dim))
    except StopIteration:
        raise Exception(f'Dimension {dim} not found in {self.__class__.__name__}.') from None