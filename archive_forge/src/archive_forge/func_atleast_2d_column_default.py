import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def atleast_2d_column_default(a, preserve_pandas=False):
    if preserve_pandas and have_pandas:
        if isinstance(a, pandas.Series):
            return pandas.DataFrame(a)
        elif isinstance(a, pandas.DataFrame):
            return a
    a = np.asarray(a)
    a = np.atleast_1d(a)
    if a.ndim <= 1:
        a = a.reshape((-1, 1))
    assert a.ndim >= 2
    return a