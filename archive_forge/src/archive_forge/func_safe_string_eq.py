import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def safe_string_eq(obj, value):
    if isinstance(obj, six.string_types):
        return obj == value
    else:
        return False