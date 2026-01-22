from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
@property
def max_x(self):
    """Return the value of the maximum X coordinate."""
    return self._envelope.MaxX