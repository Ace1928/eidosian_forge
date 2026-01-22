import numpy
from rasterio.env import GDALVersion
def is_ndarray(array):
    """Check if array is a ndarray."""
    import numpy as np
    return isinstance(array, np.ndarray) or hasattr(array, '__array__')