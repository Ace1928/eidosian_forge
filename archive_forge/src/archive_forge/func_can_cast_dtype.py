import numpy
from rasterio.env import GDALVersion
def can_cast_dtype(values, dtype):
    """Test if values can be cast to dtype without loss of information.

    Parameters
    ----------
    values: list-like
    dtype: numpy dtype or string

    Returns
    -------
    boolean
        True if values can be cast to data type.
    """
    import numpy as np
    if not is_ndarray(values):
        values = np.array(values)
    if values.dtype.name == _getnpdtype(dtype).name:
        return True
    elif values.dtype.kind == 'f':
        return np.allclose(values, values.astype(dtype), equal_nan=True)
    else:
        return np.array_equal(values, values.astype(dtype))