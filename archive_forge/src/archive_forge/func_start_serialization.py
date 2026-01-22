import json
from django.contrib.gis.gdal import CoordTransform, SpatialReference
from django.core.serializers.base import SerializerDoesNotExist
from django.core.serializers.json import Serializer as JSONSerializer
def start_serialization(self):
    self._init_options()
    self._cts = {}
    self.stream.write('{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:%d"}}, "features": [' % self.srid)