import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_get_a_service_provider(self):
    service_provider = PROVIDERS.federation_api.create_sp(uuid.uuid4().hex, unit.new_service_provider_ref())
    with self.test_client() as c:
        r = c.get('/v3/OS-FEDERATION/service_providers/%s' % service_provider['id'], headers=self.headers)
        self.assertEqual(service_provider['id'], r.json['service_provider']['id'])