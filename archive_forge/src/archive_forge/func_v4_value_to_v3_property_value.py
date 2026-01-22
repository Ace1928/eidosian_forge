from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v4_value_to_v3_property_value(self, v4_value, v3_value):
    """Converts a v4 Value to a v3 PropertyValue.

    Args:
      v4_value: an entity_v4_pb.Value
      v3_value: an entity_pb.PropertyValue to populate
    """
    v3_value.Clear()
    if v4_value.has_boolean_value():
        v3_value.set_booleanvalue(v4_value.boolean_value())
    elif v4_value.has_integer_value():
        v3_value.set_int64value(v4_value.integer_value())
    elif v4_value.has_double_value():
        v3_value.set_doublevalue(v4_value.double_value())
    elif v4_value.has_timestamp_microseconds_value():
        v3_value.set_int64value(v4_value.timestamp_microseconds_value())
    elif v4_value.has_key_value():
        v3_ref = entity_pb.Reference()
        self.v4_to_v3_reference(v4_value.key_value(), v3_ref)
        self.v3_reference_to_v3_property_value(v3_ref, v3_value)
    elif v4_value.has_blob_key_value():
        v3_value.set_stringvalue(v4_value.blob_key_value())
    elif v4_value.has_string_value():
        v3_value.set_stringvalue(v4_value.string_value())
    elif v4_value.has_blob_value():
        v3_value.set_stringvalue(v4_value.blob_value())
    elif v4_value.has_entity_value():
        v4_entity_value = v4_value.entity_value()
        v4_meaning = v4_value.meaning()
        if v4_meaning == MEANING_GEORSS_POINT or v4_meaning == MEANING_PREDEFINED_ENTITY_POINT:
            self.__v4_to_v3_point_value(v4_entity_value, v3_value.mutable_pointvalue())
        elif v4_meaning == MEANING_PREDEFINED_ENTITY_USER:
            self.v4_entity_to_v3_user_value(v4_entity_value, v3_value.mutable_uservalue())
        else:
            v3_entity_value = entity_pb.EntityProto()
            self.v4_to_v3_entity(v4_entity_value, v3_entity_value)
            v3_value.set_stringvalue(v3_entity_value.SerializePartialToString())
    elif v4_value.has_geo_point_value():
        point_value = v3_value.mutable_pointvalue()
        point_value.set_x(v4_value.geo_point_value().latitude())
        point_value.set_y(v4_value.geo_point_value().longitude())
    else:
        pass