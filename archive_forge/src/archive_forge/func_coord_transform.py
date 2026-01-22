import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import (
from django.contrib.gis.gdal.field import (
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str
def coord_transform(self):
    """Return the coordinate transformation object."""
    SpatialRefSys = self.spatial_backend.spatial_ref_sys()
    try:
        target_srs = SpatialRefSys.objects.using(self.using).get(srid=self.geo_field.srid).srs
        return CoordTransform(self.source_srs, target_srs)
    except Exception as exc:
        raise LayerMapError('Could not translate between the data source and model geometry.') from exc