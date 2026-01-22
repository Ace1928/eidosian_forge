import re
from django.conf import settings
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.db.models import GeometryField, RasterField
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db import NotSupportedError, ProgrammingError
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.db.models import Func, Value
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
from .adapter import PostGISAdapter
from .models import PostGISGeometryColumns, PostGISSpatialRefSys
from .pgraster import from_pgraster
def check_raster(self, lookup, template_params):
    spheroid = lookup.rhs_params and lookup.rhs_params[-1] == 'spheroid'
    lhs_is_raster = lookup.lhs.field.geom_type == 'RASTER'
    rhs_is_raster = isinstance(lookup.rhs, GDALRaster)
    if lookup.band_lhs is not None and lhs_is_raster:
        if not self.func:
            raise ValueError('Band indices are not allowed for this operator, it works on bbox only.')
        template_params['lhs'] = '%s, %s' % (template_params['lhs'], lookup.band_lhs)
    if lookup.band_rhs is not None and rhs_is_raster:
        if not self.func:
            raise ValueError('Band indices are not allowed for this operator, it works on bbox only.')
        template_params['rhs'] = '%s, %s' % (template_params['rhs'], lookup.band_rhs)
    if not self.raster or spheroid:
        if lhs_is_raster:
            template_params['lhs'] = 'ST_Polygon(%s)' % template_params['lhs']
        if rhs_is_raster:
            template_params['rhs'] = 'ST_Polygon(%s)' % template_params['rhs']
    elif self.raster == BILATERAL:
        if lhs_is_raster and (not rhs_is_raster):
            template_params['lhs'] = 'ST_Polygon(%s)' % template_params['lhs']
        elif rhs_is_raster and (not lhs_is_raster):
            template_params['rhs'] = 'ST_Polygon(%s)' % template_params['rhs']
    return template_params