from contextlib import contextmanager
import numpy as np
from_record_like = None
def requires_attr(attr, typ):
    if not hasattr(obj, attr):
        raise AttributeError(attr)
    if not isinstance(getattr(obj, attr), typ):
        raise AttributeError('%s must be of type %s' % (attr, typ))