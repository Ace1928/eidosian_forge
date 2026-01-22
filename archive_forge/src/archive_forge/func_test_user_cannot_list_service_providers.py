import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_list_service_providers(self):
    PROVIDERS.federation_api.create_sp(uuid.uuid4().hex, unit.new_service_provider_ref())
    with self.test_client() as c:
        c.get('/v3/OS-FEDERATION/service_providers', headers=self.headers, expected_status_code=http.client.FORBIDDEN)