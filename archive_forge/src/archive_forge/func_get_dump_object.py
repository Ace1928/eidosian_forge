import json
from django.contrib.gis.gdal import CoordTransform, SpatialReference
from django.core.serializers.base import SerializerDoesNotExist
from django.core.serializers.json import Serializer as JSONSerializer
def get_dump_object(self, obj):
    data = {'type': 'Feature', 'id': obj.pk if self.id_field is None else getattr(obj, self.id_field), 'properties': self._current}
    if (self.selected_fields is None or 'pk' in self.selected_fields) and 'pk' not in data['properties']:
        data['properties']['pk'] = obj._meta.pk.value_to_string(obj)
    if self._geometry:
        if self._geometry.srid != self.srid:
            if self._geometry.srid not in self._cts:
                srs = SpatialReference(self.srid)
                self._cts[self._geometry.srid] = CoordTransform(self._geometry.srs, srs)
            self._geometry.transform(self._cts[self._geometry.srid])
        data['geometry'] = json.loads(self._geometry.geojson)
    else:
        data['geometry'] = None
    return data