import numpy as np
from ...metaarray import MetaArray
def suggestDType(x):
    """Return a suitable dtype for x"""
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) == 0:
            raise Exception('can not determine dtype for empty list')
        x = x[0]
    if hasattr(x, 'dtype'):
        return x.dtype
    elif isinstance(x, float):
        return float
    elif isinstance(x, int):
        return int
    else:
        return object