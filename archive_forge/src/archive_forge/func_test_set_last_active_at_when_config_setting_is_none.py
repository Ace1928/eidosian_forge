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
def test_set_last_active_at_when_config_setting_is_none(self):
    self.config_fixture.config(group='security_compliance', disable_user_account_days_inactive=None)
    password = uuid.uuid4().hex
    user = self._create_user(password)
    with self.make_request():
        user_auth = PROVIDERS.identity_api.authenticate(user_id=user['id'], password=password)
    user_ref = self._get_user_ref(user_auth['id'])
    self.assertIsNone(user_ref.last_active_at)