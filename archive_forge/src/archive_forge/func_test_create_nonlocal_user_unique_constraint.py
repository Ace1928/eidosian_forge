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
def test_create_nonlocal_user_unique_constraint(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user_created = PROVIDERS.shadow_users_api.create_nonlocal_user(user)
    self.assertNotIn('password', user_created)
    self.assertEqual(user_created['id'], user['id'])
    self.assertEqual(user_created['domain_id'], user['domain_id'])
    self.assertEqual(user_created['name'], user['name'])
    new_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    new_user['name'] = user['name']
    self.assertRaises(exception.Conflict, PROVIDERS.shadow_users_api.create_nonlocal_user, new_user)