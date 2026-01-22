from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_geom_offset(result, func, cargs, offset=-1):
    """Check the geometry at the given offset in the C parameter list."""
    check_err(result)
    geom = ptr_byref(cargs, offset=offset)
    return check_geom(geom, func, cargs)