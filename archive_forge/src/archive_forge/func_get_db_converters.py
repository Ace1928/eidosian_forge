from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import Area as AreaMeasure
from django.contrib.gis.measure import Distance as DistanceMeasure
from django.db import NotSupportedError
from django.utils.functional import cached_property
def get_db_converters(self, expression):
    converters = super().get_db_converters(expression)
    if isinstance(expression.output_field, GeometryField):
        converters.append(self.get_geometry_converter(expression))
    return converters