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
def geo_db_type(self, f):
    """
        Return the database field type for the given spatial field.
        """
    if f.geom_type == 'RASTER':
        return 'raster'
    if f.dim == 3:
        geom_type = f.geom_type + 'Z'
    else:
        geom_type = f.geom_type
    if f.geography:
        if f.srid != 4326:
            raise NotSupportedError('PostGIS only supports geography columns with an SRID of 4326.')
        return 'geography(%s,%d)' % (geom_type, f.srid)
    else:
        return 'geometry(%s,%d)' % (geom_type, f.srid)