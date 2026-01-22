import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
def test_credential_limits(self):
    config_fixture_ = self.user = self.useFixture(config_fixture.Config())
    config_fixture_.config(group='credential', user_limit=4)
    self._create_credential_with_user_id(self.user_foo['id'])
    self.assertRaises(exception.CredentialLimitExceeded, self._create_credential_with_user_id, self.user_foo['id'])