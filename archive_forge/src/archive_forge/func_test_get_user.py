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
def test_get_user(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user.pop('email')
    user.pop('password')
    user_created = PROVIDERS.shadow_users_api.create_nonlocal_user(user)
    self.assertEqual(user_created['id'], user['id'])
    user_found = PROVIDERS.shadow_users_api.get_user(user_created['id'])
    self.assertCountEqual(user_created, user_found)