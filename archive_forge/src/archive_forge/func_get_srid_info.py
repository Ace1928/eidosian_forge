from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
def get_srid_info(srid, connection):
    """
    Return the units, unit name, and spheroid WKT associated with the
    given SRID from the `spatial_ref_sys` (or equivalent) spatial database
    table for the given database connection.  These results are cached.
    """
    from django.contrib.gis.gdal import SpatialReference
    global _srid_cache
    try:
        SpatialRefSys = connection.ops.spatial_ref_sys()
    except NotImplementedError:
        SpatialRefSys = None
    alias, get_srs = (connection.alias, lambda srid: SpatialRefSys.objects.using(connection.alias).get(srid=srid).srs) if SpatialRefSys else (None, SpatialReference)
    if srid not in _srid_cache[alias]:
        srs = get_srs(srid)
        units, units_name = srs.units
        _srid_cache[alias][srid] = SRIDCacheEntry(units=units, units_name=units_name, spheroid='SPHEROID["%s",%s,%s]' % (srs['spheroid'], srs.semi_major, srs.inverse_flattening), geodetic=srs.geographic)
    return _srid_cache[alias][srid]