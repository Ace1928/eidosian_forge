from ctypes import byref
from pathlib import Path
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.driver import Driver
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.layer import Layer
from django.contrib.gis.gdal.prototypes import ds as capi
from django.utils.encoding import force_bytes, force_str
@property
def layer_count(self):
    """Return the number of layers in the data source."""
    return capi.get_layer_count(self._ptr)