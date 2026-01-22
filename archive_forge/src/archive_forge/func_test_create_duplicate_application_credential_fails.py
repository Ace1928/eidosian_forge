import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_create_duplicate_application_credential_fails(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    name = app_cred['name']
    self.app_cred_api.create_application_credential(app_cred)
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'], name=name)
    self.assertRaises(exception.Conflict, self.app_cred_api.create_application_credential, app_cred)