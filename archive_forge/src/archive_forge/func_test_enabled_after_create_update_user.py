import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
def test_enabled_after_create_update_user(self):
    self.config_fixture.config(group='security_compliance', disable_user_account_days_inactive=90)
    del self.user_dict['enabled']
    user = PROVIDERS.identity_api.create_user(self.user_dict)
    user_ref = self._get_user_ref(user['id'])
    self.assertTrue(user_ref.enabled)
    now = datetime.datetime.utcnow().date()
    self.assertGreaterEqual(now, user_ref.last_active_at)
    user['enabled'] = True
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = self._get_user_ref(user['id'])
    self.assertTrue(user_ref.enabled)
    user['enabled'] = False
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = self._get_user_ref(user['id'])
    self.assertFalse(user_ref.enabled)
    user['enabled'] = True
    PROVIDERS.identity_api.update_user(user['id'], user)
    user_ref = self._get_user_ref(user['id'])
    self.assertTrue(user_ref.enabled)