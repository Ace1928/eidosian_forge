import datetime
from testtools import matchers
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_application_credential_with_application_credential(self):
    with self.test_client() as c:
        roles = [{'id': self.role_id}]
        app_cred_body = self._app_cred_body(roles=roles)
        token = self.get_scoped_token()
        app_cred = c.post('/v3/users/%s/application_credentials' % self.user_id, json=app_cred_body, expected_status_code=http.client.CREATED, headers={'X-Auth-Token': token})
        auth_data = self.build_authentication_request(app_cred_id=app_cred.json['application_credential']['id'], secret=app_cred.json['application_credential']['secret'])
        token_data = self.v3_create_token(auth_data, expected_status=http.client.CREATED)
        member_path = '/v3%s' % MEMBER_PATH_FMT % {'user_id': self.user_id, 'app_cred_id': app_cred.json['application_credential']['id']}
        token = token_data.headers['x-subject-token']
        c.delete(member_path, json=app_cred_body, expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': token})