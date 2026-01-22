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
def dimension_name(dimension):
    """Return the Dimension.name for a dimension-like object.

    Args:
        dimension: Dimension or dimension string, tuple or dict

    Returns:
        The name of the Dimension or what would be the name if the
        input as converted to a Dimension.
    """
    if isinstance(dimension, Dimension):
        return dimension.name
    elif isinstance(dimension, str):
        return dimension
    elif isinstance(dimension, tuple):
        return dimension[0]
    elif isinstance(dimension, dict):
        return dimension['name']
    elif dimension is None:
        return None
    else:
        raise ValueError('%s type could not be interpreted as Dimension. Dimensions must be declared as a string, tuple, dictionary or Dimension type.' % type(dimension).__name__)