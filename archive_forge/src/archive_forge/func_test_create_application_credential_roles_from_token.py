import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_roles_from_token(self):
    with self.test_client() as c:
        app_cred_body = self._app_cred_body()
        token = self.get_scoped_token()
        resp = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        self.assertThat(resp.json['application_credential']['roles'], matchers.HasLength(1))
        self.assertEqual(resp.json['application_credential']['roles'][0]['id'], self.role_id)