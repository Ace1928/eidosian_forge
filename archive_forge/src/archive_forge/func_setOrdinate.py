from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
def setOrdinate(self, dimension, index, value):
    """Set the value for the given dimension and index."""
    self._checkindex(index)
    self._checkdim(dimension)
    capi.cs_setordinate(self.ptr, index, dimension, value)