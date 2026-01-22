from __future__ import division, print_function, absolute_import
from petl.compat import next, string_types
from petl.util.base import iterpeek, ValuesView, Table
from petl.util.materialise import columns
def torecarray(*args, **kwargs):
    """
    Convenient shorthand for ``toarray(*args, **kwargs).view(np.recarray)``.

    """
    import numpy as np
    return toarray(*args, **kwargs).view(np.recarray)