import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_create_application_credential_with_access_rules(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    app_cred['access_rules'] = [{'id': uuid.uuid4().hex, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]
    resp = self.app_cred_api.create_application_credential(app_cred)
    resp.pop('roles')
    resp_access_rules = resp.pop('access_rules')
    app_cred.pop('roles')
    orig_access_rules = app_cred.pop('access_rules')
    self.assertDictEqual(app_cred, resp)
    for i, ar in enumerate(resp_access_rules):
        self.assertDictEqual(orig_access_rules[i], ar)