import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_implied_roles(self):
    PROVIDERS.role_api.create_implied_role(self.prior_role_id, self.implied_role_id)
    with self.test_client() as c:
        r = c.get('/v3/roles/%s/implies' % self.prior_role_id, headers=self.headers)
        self.assertEqual(1, len(r.json['role_inference']['implies']))