import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_validate_their_own_tokens(self):
    with self.test_client() as c:
        self.headers['X-Subject-Token'] = self.token_id
        c.get('/v3/auth/tokens', headers=self.headers)