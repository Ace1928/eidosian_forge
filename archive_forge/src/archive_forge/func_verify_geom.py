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
def verify_geom(self, geom, model_field):
    """
        Verify the geometry -- construct and return a GeometryCollection
        if necessary (for example if the model field is MultiPolygonField while
        the mapped shapefile only contains Polygons).
        """
    if self.coord_dim != geom.coord_dim:
        geom.coord_dim = self.coord_dim
    if self.make_multi(geom.geom_type, model_field):
        multi_type = self.MULTI_TYPES[geom.geom_type.num]
        g = OGRGeometry(multi_type)
        g.add(geom)
    else:
        g = geom
    if self.transform:
        g.transform(self.transform)
    return g.wkt