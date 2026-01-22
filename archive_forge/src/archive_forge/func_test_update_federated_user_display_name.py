import datetime
from unittest import mock
import uuid
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import sql as shadow_sql
from keystone.tests import unit
def test_update_federated_user_display_name(self):
    user_dict_create = PROVIDERS.shadow_users_api.create_federated_user(self.domain_id, self.federated_user)
    new_display_name = uuid.uuid4().hex
    PROVIDERS.shadow_users_api.update_federated_user_display_name(self.federated_user['idp_id'], self.federated_user['protocol_id'], self.federated_user['unique_id'], new_display_name)
    user_ref = PROVIDERS.shadow_users_api._get_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], self.federated_user['unique_id'])
    self.assertEqual(user_ref.federated_users[0].display_name, new_display_name)
    self.assertEqual(user_dict_create['id'], user_ref.id)