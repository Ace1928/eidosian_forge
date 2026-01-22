import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_get_user_returns_required_attributes(self):
    user_ref = PROVIDERS.identity_api.get_user(self.user_foo['id'])
    self.assertIn('id', user_ref)
    self.assertIn('name', user_ref)
    self.assertIn('enabled', user_ref)
    self.assertIn('password_expires_at', user_ref)