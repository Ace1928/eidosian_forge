import logging
import os
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_char_p
from ctypes.util import find_library
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import SimpleLazyObject, cached_property
from django.utils.version import get_version_tuple
def notice_h(fmt, lst):
    fmt, lst = (fmt.decode(), lst.decode())
    try:
        warn_msg = fmt % lst
    except TypeError:
        warn_msg = fmt
    logger.warning('GEOS_NOTICE: %s\n', warn_msg)