import logging
import os
import re
from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int
from ctypes.util import find_library
from django.contrib.gis.gdal.error import GDALException
from django.core.exceptions import ImproperlyConfigured
def std_call(func):
    """
    Return the correct STDCALL function for certain OSR routines on Win32
    platforms.
    """
    if os.name == 'nt':
        return lwingdal[func]
    else:
        return lgdal[func]