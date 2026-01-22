import logging
import os
import re
from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int
from ctypes.util import find_library
from django.contrib.gis.gdal.error import GDALException
from django.core.exceptions import ImproperlyConfigured
def gdal_version_info():
    ver = gdal_version()
    m = re.match(b'^(?P<major>\\d+)\\.(?P<minor>\\d+)(?:\\.(?P<subminor>\\d+))?', ver)
    if not m:
        raise GDALException('Could not parse GDAL version string "%s"' % ver)
    major, minor, subminor = m.groups()
    return (int(major), int(minor), subminor and int(subminor))