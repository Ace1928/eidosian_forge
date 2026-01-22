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
def test_password_not_expired_for_ignore_user(self):
    initial_password = uuid.uuid4().hex
    user = self._create_user(initial_password, False)
    self.assertPasswordIsNotExpired(user['id'], initial_password)
    self.config_fixture.config(group='security_compliance', change_password_upon_first_use=True)
    user['options'][iro.IGNORE_CHANGE_PASSWORD_OPT.option_name] = True
    admin_password = uuid.uuid4().hex
    user['password'] = admin_password
    PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertPasswordIsNotExpired(user['id'], admin_password)
    user['options'][iro.IGNORE_CHANGE_PASSWORD_OPT.option_name] = False
    admin_password = uuid.uuid4().hex
    user['password'] = admin_password
    PROVIDERS.identity_api.update_user(user['id'], user)
    self.assertPasswordIsExpired(user['id'], admin_password)