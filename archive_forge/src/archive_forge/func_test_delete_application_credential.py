import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_delete_application_credential(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    self.app_cred_api.create_application_credential(app_cred)
    self.app_cred_api.get_application_credential(app_cred['id'])
    self.assertIn(app_cred['id'], self._list_ids(self.user_foo))
    self.app_cred_api.delete_application_credential(app_cred['id'])
    self.assertNotIn(app_cred['id'], self._list_ids(self.user_foo))
    self.assertRaises(exception.ApplicationCredentialNotFound, self.app_cred_api.get_application_credential, app_cred['id'])