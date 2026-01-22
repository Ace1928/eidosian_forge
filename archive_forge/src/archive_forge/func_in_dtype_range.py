import numpy
from rasterio.env import GDALVersion
def in_dtype_range(value, dtype):
    """Test if the value is within the dtype's range of values, Nan, or Inf."""
    kind = numpy.dtype(dtype).kind
    if kind == 'f' and (numpy.isnan(value) or numpy.isinf(value)):
        return True
    info = dtype_info_registry[kind](dtype)
    return info.min <= value <= info.max