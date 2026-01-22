from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_to_v3_property(self, property_name, is_multi, is_projection, v1_value, v3_property):
    """Converts info from a v1 Property to a v3 Property.

    v1_value must not have an array_value.

    Args:
      property_name: the name of the property, unicode
      is_multi: whether the property contains multiple values
      is_projection: whether the property is projected
      v1_value: an googledatastore.Value
      v3_property: an entity_pb.Property to populate
    """
    v1_value_type = v1_value.WhichOneof('value_type')
    if v1_value_type == 'array_value':
        assert False, 'v1 array_value not convertable to v3'
    v3_property.Clear()
    v3_property.set_name(property_name.encode('utf-8'))
    v3_property.set_multiple(is_multi)
    self.v1_value_to_v3_property_value(v1_value, v3_property.mutable_value())
    v1_meaning = None
    if v1_value.meaning:
        v1_meaning = v1_value.meaning
    if v1_value_type == 'timestamp_value':
        v3_property.set_meaning(entity_pb.Property.GD_WHEN)
    elif v1_value_type == 'blob_value':
        if v1_meaning == MEANING_ZLIB:
            v3_property.set_meaning_uri(URI_MEANING_ZLIB)
        if v1_meaning == entity_pb.Property.BYTESTRING:
            if not v1_value.exclude_from_indexes:
                pass
        else:
            if not v1_value.exclude_from_indexes:
                v3_property.set_meaning(entity_pb.Property.BYTESTRING)
            else:
                v3_property.set_meaning(entity_pb.Property.BLOB)
            v1_meaning = None
    elif v1_value_type == 'entity_value':
        if v1_meaning != MEANING_PREDEFINED_ENTITY_USER:
            v3_property.set_meaning(entity_pb.Property.ENTITY_PROTO)
        v1_meaning = None
    elif v1_value_type == 'geo_point_value':
        if v1_meaning != MEANING_POINT_WITHOUT_V3_MEANING:
            v3_property.set_meaning(MEANING_GEORSS_POINT)
        v1_meaning = None
    elif v1_value_type == 'integer_value':
        if v1_meaning == MEANING_NON_RFC_3339_TIMESTAMP:
            v3_property.set_meaning(entity_pb.Property.GD_WHEN)
            v1_meaning = None
    else:
        pass
    if v1_meaning is not None:
        v3_property.set_meaning(v1_meaning)
    if is_projection:
        v3_property.set_meaning(entity_pb.Property.INDEX_VALUE)