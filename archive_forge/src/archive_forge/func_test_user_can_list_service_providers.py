import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_list_service_providers(self):
    service_provider = PROVIDERS.federation_api.create_sp(uuid.uuid4().hex, unit.new_service_provider_ref())
    with self.test_client() as c:
        r = c.get('/v3/OS-FEDERATION/service_providers', headers=self.headers)
        self.assertEqual(1, len(r.json['service_providers']))
        self.assertEqual(service_provider['id'], r.json['service_providers'][0]['id'])