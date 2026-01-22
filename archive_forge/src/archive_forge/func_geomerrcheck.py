from ctypes import POINTER, c_bool, c_char_p, c_double, c_int, c_int64, c_void_p
from functools import partial
from django.contrib.gis.gdal.prototypes.errcheck import (
def geomerrcheck(result, func, cargs):
    return check_geom_offset(result, func, cargs, offset)