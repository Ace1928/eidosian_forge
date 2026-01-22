from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v4_to_v3_entity(self, v4_entity, v3_entity, is_projection=False):
    """Converts a v4 Entity to a v3 EntityProto.

    Args:
      v4_entity: an entity_v4_pb.Entity
      v3_entity: an entity_pb.EntityProto to populate
      is_projection: True if the v4_entity is from a projection query.
    """
    v3_entity.Clear()
    for v4_property in v4_entity.property_list():
        property_name = v4_property.name()
        v4_value = v4_property.value()
        if v4_value.list_value_list():
            for v4_sub_value in v4_value.list_value_list():
                self.__add_v3_property_from_v4(property_name, True, is_projection, v4_sub_value, v3_entity)
        else:
            self.__add_v3_property_from_v4(property_name, False, is_projection, v4_value, v3_entity)
    if v4_entity.has_key():
        v4_key = v4_entity.key()
        self.v4_to_v3_reference(v4_key, v3_entity.mutable_key())
        v3_ref = v3_entity.key()
        self.v3_reference_to_group(v3_ref, v3_entity.mutable_entity_group())
    else:
        pass