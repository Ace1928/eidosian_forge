from ctypes import c_void_p, string_at
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOSFuncFactory
def last_arg_byref(args):
    """Return the last C argument's value by reference."""
    return args[-1]._obj.value