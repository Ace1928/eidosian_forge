import numpy as np
from ..sharing import to_backend_cache_wrap
@to_backend_cache_wrap
def to_cupy(array):
    import cupy
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    return array