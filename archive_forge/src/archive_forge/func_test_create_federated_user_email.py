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
def test_create_federated_user_email(self):
    user = PROVIDERS.shadow_users_api.create_federated_user(self.domain_id, self.federated_user, self.email)
    self.assertEqual(user['email'], self.email)