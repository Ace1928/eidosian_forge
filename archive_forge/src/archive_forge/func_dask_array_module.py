import numpy as np
from .. import util
def dask_array_module():
    try:
        import dask.array as da
        return da
    except ImportError:
        return None