from django.contrib.gis.db.backends.base.features import BaseSpatialFeatures
from django.db.backends.sqlite3.features import (
from django.utils.functional import cached_property
@cached_property
def supports_area_geodetic(self):
    return bool(self.connection.ops.geom_lib_version())