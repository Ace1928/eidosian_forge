import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def safe_scalar_isnan(x):
    try:
        return np.isnan(float(x))
    except (TypeError, ValueError, NotImplementedError):
        return False