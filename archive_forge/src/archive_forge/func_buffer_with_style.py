import re
from ctypes import addressof, byref, c_double
from django.contrib.gis import gdal
from django.contrib.gis.geometry import hex_regex, json_regex, wkt_regex
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOM_PTR, geos_version_tuple
from django.contrib.gis.geos.mutable_list import ListMixin
from django.contrib.gis.geos.prepared import PreparedGeometry
from django.contrib.gis.geos.prototypes.io import ewkb_w, wkb_r, wkb_w, wkt_r, wkt_w
from django.utils.deconstruct import deconstructible
from django.utils.encoding import force_bytes, force_str
def buffer_with_style(self, width, quadsegs=8, end_cap_style=1, join_style=1, mitre_limit=5.0):
    """
        Same as buffer() but allows customizing the style of the memoryview.

        End cap style can be round (1), flat (2), or square (3).
        Join style can be round (1), mitre (2), or bevel (3).
        Mitre ratio limit only affects mitered join style.
        """
    return self._topology(capi.geos_bufferwithstyle(self.ptr, width, quadsegs, end_cap_style, join_style, mitre_limit))