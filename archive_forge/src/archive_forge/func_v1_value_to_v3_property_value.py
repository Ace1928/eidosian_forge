from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_value_to_v3_property_value(self, v1_value, v3_value):
    """Converts a v1 Value to a v3 PropertyValue.

    Args:
      v1_value: an googledatastore.Value
      v3_value: an entity_pb.PropertyValue to populate
    """
    v3_value.Clear()
    field = v1_value.WhichOneof('value_type')
    if field == 'boolean_value':
        v3_value.set_booleanvalue(v1_value.boolean_value)
    elif field == 'integer_value':
        v3_value.set_int64value(v1_value.integer_value)
    elif field == 'double_value':
        v3_value.set_doublevalue(v1_value.double_value)
    elif field == 'timestamp_value':
        v3_value.set_int64value(googledatastore.helper.micros_from_timestamp(v1_value.timestamp_value))
    elif field == 'key_value':
        v3_ref = entity_pb.Reference()
        self.v1_to_v3_reference(v1_value.key_value, v3_ref)
        self.v3_reference_to_v3_property_value(v3_ref, v3_value)
    elif field == 'string_value':
        v3_value.set_stringvalue(v1_value.string_value.encode('utf-8'))
    elif field == 'blob_value':
        v3_value.set_stringvalue(v1_value.blob_value)
    elif field == 'entity_value':
        v1_entity_value = v1_value.entity_value
        v1_meaning = v1_value.meaning
        if v1_meaning == MEANING_PREDEFINED_ENTITY_USER:
            self.v1_entity_to_v3_user_value(v1_entity_value, v3_value.mutable_uservalue())
        else:
            v3_entity_value = entity_pb.EntityProto()
            self.v1_to_v3_entity(v1_entity_value, v3_entity_value)
            v3_value.set_stringvalue(v3_entity_value.SerializePartialToString())
    elif field == 'geo_point_value':
        point_value = v3_value.mutable_pointvalue()
        point_value.set_x(v1_value.geo_point_value.latitude)
        point_value.set_y(v1_value.geo_point_value.longitude)
    elif field == 'null_value':
        pass
    else:
        pass