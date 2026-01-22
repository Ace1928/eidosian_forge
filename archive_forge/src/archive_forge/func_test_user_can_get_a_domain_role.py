import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_a_domain_role(self):
    role = PROVIDERS.role_api.create_role(uuid.uuid4().hex, unit.new_role_ref(domain_id=CONF.identity.default_domain_id))
    with self.test_client() as c:
        r = c.get('/v3/roles/%s' % role['id'], headers=self.headers)
        self.assertEqual(role['id'], r.json['role']['id'])