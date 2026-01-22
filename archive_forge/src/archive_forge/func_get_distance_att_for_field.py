from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import Area as AreaMeasure
from django.contrib.gis.measure import Distance as DistanceMeasure
from django.db import NotSupportedError
from django.utils.functional import cached_property
def get_distance_att_for_field(self, field):
    dist_att = None
    if field.geodetic(self.connection):
        if self.connection.features.supports_distance_geodetic:
            dist_att = 'm'
    else:
        units = field.units_name(self.connection)
        if units:
            dist_att = DistanceMeasure.unit_attname(units)
    return dist_att