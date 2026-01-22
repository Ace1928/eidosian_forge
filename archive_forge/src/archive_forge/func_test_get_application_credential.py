import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_get_application_credential(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    create_resp = self.app_cred_api.create_application_credential(app_cred)
    app_cred_id = create_resp['id']
    get_resp = self.app_cred_api.get_application_credential(app_cred_id)
    create_resp.pop('secret')
    self.assertDictEqual(create_resp, get_resp)