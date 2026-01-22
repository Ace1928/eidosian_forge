import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_limits(self):
    project_limit_id, _ = _create_limits_and_dependencies()
    limit = PROVIDERS.unified_limit_api.get_limit(project_limit_id)
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=self.user_id, project_id=limit['project_id'])
    auth = self.build_authentication_request(user_id=self.user_id, password=self.bootstrapper.admin_password, project_id=limit['project_id'])
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=auth)
        token_id = r.headers['X-Subject-Token']
        headers = {'X-Auth-Token': token_id}
    with self.test_client() as c:
        r = c.get('/v3/limits', headers=headers)
        self.assertTrue(len(r.json['limits']) == 1)
        self.assertEqual(project_limit_id, r.json['limits'][0]['id'])