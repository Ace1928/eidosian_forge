from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_pointer(result, func, cargs):
    """Make sure the result pointer is valid."""
    if isinstance(result, int):
        result = c_void_p(result)
    if result:
        return result
    else:
        raise GDALException('Invalid pointer returned from "%s"' % func.__name__)