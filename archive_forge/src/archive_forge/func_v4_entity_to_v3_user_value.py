from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v4_entity_to_v3_user_value(self, v4_user_entity, v3_user_value):
    """Converts a v4 user Entity to a v3 UserValue.

    Args:
      v4_user_entity: an entity_v4_pb.Entity representing a user
      v3_user_value: an entity_pb.Property_UserValue to populate
    """
    v3_user_value.Clear()
    name_to_v4_property = self.__build_name_to_v4_property_map(v4_user_entity)
    v3_user_value.set_email(self.__get_v4_string_value(name_to_v4_property[PROPERTY_NAME_EMAIL]))
    v3_user_value.set_auth_domain(self.__get_v4_string_value(name_to_v4_property[PROPERTY_NAME_AUTH_DOMAIN]))
    if PROPERTY_NAME_USER_ID in name_to_v4_property:
        v3_user_value.set_obfuscated_gaiaid(self.__get_v4_string_value(name_to_v4_property[PROPERTY_NAME_USER_ID]))
    if PROPERTY_NAME_INTERNAL_ID in name_to_v4_property:
        v3_user_value.set_gaiaid(self.__get_v4_integer_value(name_to_v4_property[PROPERTY_NAME_INTERNAL_ID]))
    else:
        v3_user_value.set_gaiaid(0)
    if PROPERTY_NAME_FEDERATED_IDENTITY in name_to_v4_property:
        v3_user_value.set_federated_identity(self.__get_v4_string_value(name_to_v4_property[PROPERTY_NAME_FEDERATED_IDENTITY]))
    if PROPERTY_NAME_FEDERATED_PROVIDER in name_to_v4_property:
        v3_user_value.set_federated_provider(self.__get_v4_string_value(name_to_v4_property[PROPERTY_NAME_FEDERATED_PROVIDER]))