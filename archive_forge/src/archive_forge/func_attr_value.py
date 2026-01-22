from ctypes import byref, c_char_p, c_int
from enum import IntEnum
from types import NoneType
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import SRSException
from django.contrib.gis.gdal.libgdal import GDAL_VERSION
from django.contrib.gis.gdal.prototypes import srs as capi
from django.utils.encoding import force_bytes, force_str
def attr_value(self, target, index=0):
    """
        The attribute value for the given target node (e.g. 'PROJCS'). The index
        keyword specifies an index of the child node to return.
        """
    if not isinstance(target, str) or not isinstance(index, int):
        raise TypeError
    return capi.get_attr_value(self.ptr, force_bytes(target), index)