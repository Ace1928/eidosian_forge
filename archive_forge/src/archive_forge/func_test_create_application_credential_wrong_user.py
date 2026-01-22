import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_create_application_credential_wrong_user(self):
    wrong_user = unit.create_user(PROVIDERS.identity_api, test_v3.DEFAULT_DOMAIN_ID)
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        c.post('/v3/users/%s/application_credentials' % wrong_user['id'], json=app_cred_body, expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': token})