import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_user_cannot_lookup_application_credential_for_another_user(self):
    another_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    another_user_id = PROVIDERS.identity_api.create_user(another_user)['id']
    auth = self.build_authentication_request(user_id=another_user_id, password=another_user['password'])
    with self.test_client() as c:
        r = c.post('/v3/auth/tokens', json=auth)
        another_user_token = r.headers['X-Subject-Token']
    app_cred = self._create_application_credential()
    with self.test_client() as c:
        c.get('/v3/users/%s/application_credentials/%s' % (another_user_id, app_cred['id']), expected_status_code=http.client.FORBIDDEN, headers={'X-Auth-Token': another_user_token})