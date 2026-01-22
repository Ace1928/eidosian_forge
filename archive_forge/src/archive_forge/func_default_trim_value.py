import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
def default_trim_value():
    """
    GEOS changed the default value in 3.12.0. Can be replaced by True when
    3.12.0 becomes the minimum supported version.
    """
    return geos_version_tuple() >= (3, 12)