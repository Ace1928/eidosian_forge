from ctypes import byref, c_char_p, c_int
from enum import IntEnum
from types import NoneType
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import SRSException
from django.contrib.gis.gdal.libgdal import GDAL_VERSION
from django.contrib.gis.gdal.prototypes import srs as capi
from django.utils.encoding import force_bytes, force_str
def identify_epsg(self):
    """
        This method inspects the WKT of this SpatialReference, and will
        add EPSG authority nodes where an EPSG identifier is applicable.
        """
    capi.identify_epsg(self.ptr)