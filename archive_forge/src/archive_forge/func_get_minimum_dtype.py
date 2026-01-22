import numpy
from rasterio.env import GDALVersion
def get_minimum_dtype(values):
    """Determine minimum type to represent values.

    Uses range checking to determine the minimum integer or floating point
    data type required to represent values.

    Parameters
    ----------
    values: list-like


    Returns
    -------
    rasterio dtype string
    """
    import numpy as np
    if not is_ndarray(values):
        values = np.array(values)
    min_value = values.min()
    max_value = values.max()
    if values.dtype.kind in ('i', 'u'):
        if min_value >= 0:
            if max_value <= 255:
                return uint8
            elif max_value <= 65535:
                return uint16
            elif max_value <= 4294967295:
                return uint32
            if not _GDAL_AT_LEAST_35:
                raise ValueError('Values out of range for supported dtypes')
            return uint64
        elif min_value >= -32768 and max_value <= 32767:
            return int16
        elif min_value >= -2147483648 and max_value <= 2147483647:
            return int32
        if not _GDAL_AT_LEAST_35:
            raise ValueError('Values out of range for supported dtypes')
        return int64
    else:
        if min_value >= -3.4028235e+38 and max_value <= 3.4028235e+38:
            return float32
        return float64