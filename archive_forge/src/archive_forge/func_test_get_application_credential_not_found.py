import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_get_application_credential_not_found(self):
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, uuid.uuid4().hex)