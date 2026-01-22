import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_with_duplicate_access_rule(self):
    roles = [{'id': self.role_id}]
    access_rules = [{'path': '/v3/projects', 'method': 'POST', 'service': 'identity'}]
    app_cred_body_1 = self._app_cred_body(roles=roles, access_rules=access_rules)
    with self.test_client() as c:
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, headers={'X-Auth-Token': token}, json=app_cred_body_1, expected_status_code=http.client.CREATED)
    resp_access_rules = resp.json['application_credential']['access_rules']
    self.assertIn('id', resp_access_rules[0])
    access_rule_id = resp_access_rules[0].pop('id')
    self.assertEqual(access_rules[0], resp_access_rules[0])
    app_cred_body_2 = self._app_cred_body(roles=roles, access_rules=access_rules)
    with self.test_client() as c:
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, headers={'X-Auth-Token': token}, json=app_cred_body_2, expected_status_code=http.client.CREATED)
    resp_access_rules = resp.json['application_credential']['access_rules']
    self.assertEqual(access_rule_id, resp_access_rules[0]['id'])