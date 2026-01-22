import threading
from ctypes import POINTER, Structure, byref, c_byte, c_char_p, c_int, c_size_t
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.libgeos import (
from django.contrib.gis.geos.prototypes.errcheck import (
from django.contrib.gis.geos.prototypes.geom import c_uchar_p, geos_char_p
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
@outdim.setter
def outdim(self, new_dim):
    if new_dim not in (2, 3):
        raise ValueError('WKB output dimension must be 2 or 3')
    wkb_writer_set_outdim(self.ptr, new_dim)