from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v4_to_v3_property(self, property_name, is_multi, is_projection, v4_value, v3_property):
    """Converts info from a v4 Property to a v3 Property.

    v4_value must not have a list_value.

    Args:
      property_name: the name of the property
      is_multi: whether the property contains multiple values
      is_projection: whether the property is projected
      v4_value: an entity_v4_pb.Value
      v3_property: an entity_pb.Property to populate
    """
    assert not v4_value.list_value_list(), 'v4 list_value not convertable to v3'
    v3_property.Clear()
    v3_property.set_name(property_name)
    if v4_value.has_meaning() and v4_value.meaning() == MEANING_EMPTY_LIST:
        v3_property.set_meaning(MEANING_EMPTY_LIST)
        v3_property.set_multiple(False)
        v3_property.mutable_value()
        return
    v3_property.set_multiple(is_multi)
    self.v4_value_to_v3_property_value(v4_value, v3_property.mutable_value())
    v4_meaning = None
    if v4_value.has_meaning():
        v4_meaning = v4_value.meaning()
    if v4_value.has_timestamp_microseconds_value():
        v3_property.set_meaning(entity_pb.Property.GD_WHEN)
    elif v4_value.has_blob_key_value():
        v3_property.set_meaning(entity_pb.Property.BLOBKEY)
    elif v4_value.has_blob_value():
        if v4_meaning == MEANING_ZLIB:
            v3_property.set_meaning_uri(URI_MEANING_ZLIB)
        if v4_meaning == entity_pb.Property.BYTESTRING:
            if v4_value.indexed():
                pass
        else:
            if v4_value.indexed():
                v3_property.set_meaning(entity_pb.Property.BYTESTRING)
            else:
                v3_property.set_meaning(entity_pb.Property.BLOB)
            v4_meaning = None
    elif v4_value.has_entity_value():
        if v4_meaning != MEANING_GEORSS_POINT:
            if v4_meaning != MEANING_PREDEFINED_ENTITY_POINT and v4_meaning != MEANING_PREDEFINED_ENTITY_USER:
                v3_property.set_meaning(entity_pb.Property.ENTITY_PROTO)
            v4_meaning = None
    elif v4_value.has_geo_point_value():
        v3_property.set_meaning(MEANING_GEORSS_POINT)
    else:
        pass
    if v4_meaning is not None:
        v3_property.set_meaning(v4_meaning)
    if is_projection:
        v3_property.set_meaning(entity_pb.Property.INDEX_VALUE)