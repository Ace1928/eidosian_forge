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
def test_authenticate_user_disabled_due_to_inactivity(self):
    last_active_at = datetime.datetime.utcnow() - datetime.timedelta(days=self.max_inactive_days + 1)
    user = self._create_user(self.user_dict, last_active_at.date())
    with self.make_request():
        self.assertRaises(exception.UserDisabled, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=self.password)
        user = PROVIDERS.identity_api.get_user(user['id'])
        self.assertFalse(user['enabled'])
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=self.password)
        self.assertTrue(user['enabled'])