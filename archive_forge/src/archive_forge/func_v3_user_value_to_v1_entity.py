from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_user_value_to_v1_entity(self, v3_user_value, v1_entity):
    """Converts a v3 UserValue to a v1 user Entity.

    Args:
      v3_user_value: an entity_pb.Property_UserValue
      v1_entity: an googledatastore.Entity to populate
    """
    v1_entity.Clear()
    self.__v1_string_property(v1_entity, PROPERTY_NAME_EMAIL, v3_user_value.email(), False)
    self.__v1_string_property(v1_entity, PROPERTY_NAME_AUTH_DOMAIN, v3_user_value.auth_domain(), False)
    if v3_user_value.gaiaid() != 0:
        self.__v1_integer_property(v1_entity, PROPERTY_NAME_INTERNAL_ID, v3_user_value.gaiaid(), False)
    if v3_user_value.has_obfuscated_gaiaid():
        self.__v1_string_property(v1_entity, PROPERTY_NAME_USER_ID, v3_user_value.obfuscated_gaiaid(), False)
    if v3_user_value.has_federated_identity():
        self.__v1_string_property(v1_entity, PROPERTY_NAME_FEDERATED_IDENTITY, v3_user_value.federated_identity(), False)
    if v3_user_value.has_federated_provider():
        self.__v1_string_property(v1_entity, PROPERTY_NAME_FEDERATED_PROVIDER, v3_user_value.federated_provider(), False)