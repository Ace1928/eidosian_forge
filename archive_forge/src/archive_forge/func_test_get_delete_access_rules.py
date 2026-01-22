import datetime
import uuid
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def test_get_delete_access_rules(self):
    app_cred = self._new_app_cred_data(self.user_foo['id'], project_id=self.project_bar['id'])
    access_rule_id = uuid.uuid4().hex
    app_cred['access_rules'] = [{'id': access_rule_id, 'service': uuid.uuid4().hex, 'path': uuid.uuid4().hex, 'method': uuid.uuid4().hex[16:]}]
    self.app_cred_api.create_application_credential(app_cred)
    self.assertDictEqual(app_cred['access_rules'][0], self.app_cred_api.get_access_rule(access_rule_id))
    self.app_cred_api.delete_application_credential(app_cred['id'])
    self.app_cred_api.delete_access_rule(access_rule_id)
    self.assertRaises(exception.AccessRuleNotFound, self.app_cred_api.get_access_rule, access_rule_id)