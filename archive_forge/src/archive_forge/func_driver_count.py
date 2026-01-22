from ctypes import c_void_p
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as vcapi
from django.contrib.gis.gdal.prototypes import raster as rcapi
from django.utils.encoding import force_bytes, force_str
@classmethod
def driver_count(cls):
    """
        Return the number of GDAL/OGR data source drivers registered.
        """
    return vcapi.get_driver_count() + rcapi.get_driver_count()