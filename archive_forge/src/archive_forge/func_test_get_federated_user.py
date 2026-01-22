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
def test_get_federated_user(self):
    user_dict_create = PROVIDERS.shadow_users_api.create_federated_user(self.domain_id, self.federated_user)
    user_dict_get = PROVIDERS.shadow_users_api.get_federated_user(self.federated_user['idp_id'], self.federated_user['protocol_id'], self.federated_user['unique_id'])
    self.assertCountEqual(user_dict_create, user_dict_get)
    self.assertEqual(user_dict_create['id'], user_dict_get['id'])