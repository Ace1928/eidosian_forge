from ctypes import byref, c_char_p, c_int
from enum import IntEnum
from types import NoneType
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import SRSException
from django.contrib.gis.gdal.libgdal import GDAL_VERSION
from django.contrib.gis.gdal.prototypes import srs as capi
from django.utils.encoding import force_bytes, force_str
@property
def pretty_wkt(self, simplify=0):
    """Return the 'pretty' representation of the WKT."""
    return capi.to_pretty_wkt(self.ptr, byref(c_char_p()), simplify)