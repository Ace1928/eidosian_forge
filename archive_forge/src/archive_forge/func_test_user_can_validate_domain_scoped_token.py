import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_validate_domain_scoped_token(self):
    domain = PROVIDERS.resource_api.create_domain(uuid.uuid4().hex, unit.new_domain_ref())
    user = unit.new_user_ref(domain_id=domain['id'])
    user['id'] = PROVIDERS.identity_api.create_user(user)['id']
    PROVIDERS.assignment_api.create_grant(self.bootstrapper.reader_role_id, user_id=user['id'], domain_id=domain['id'])
    domain_auth = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=domain['id'])
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=domain_auth)
        domain_token = r.headers['X-Subject-Token']
    with self.test_client() as c:
        self.headers['X-Subject-Token'] = domain_token
        c.get('/v3/auth/tokens', headers=self.headers)