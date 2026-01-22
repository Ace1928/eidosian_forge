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
def test_user_password_created_expired_at_int_matches_created_at(self):
    with sql.session_for_read() as session:
        user_ref = PROVIDERS.identity_api._get_user(session, self.user_foo['id'])
        self.assertIsNotNone(user_ref.password_ref._created_at)
        self.assertIsNotNone(user_ref.password_ref._expires_at)
        self.assertEqual(user_ref.password_ref._created_at, user_ref.password_ref.created_at_int)
        self.assertEqual(user_ref.password_ref._expires_at, user_ref.password_ref.expires_at_int)
        self.assertEqual(user_ref.password_ref.created_at, user_ref.password_ref.created_at_int)
        self.assertEqual(user_ref.password_ref.expires_at, user_ref.password_ref.expires_at_int)