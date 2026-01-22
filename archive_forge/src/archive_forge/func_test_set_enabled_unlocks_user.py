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
def test_set_enabled_unlocks_user(self):
    with self.make_request():
        self._fail_auth_repeatedly(self.user['id'])
        self.assertRaises(exception.Unauthorized, PROVIDERS.identity_api.authenticate, user_id=self.user['id'], password=uuid.uuid4().hex)
        self.user['enabled'] = True
        PROVIDERS.identity_api.update_user(self.user['id'], self.user)
        user_ret = PROVIDERS.identity_api.authenticate(user_id=self.user['id'], password=self.password)
        self.assertTrue(user_ret['enabled'])