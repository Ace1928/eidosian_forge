import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_delete_services(self):
    service = unit.new_service_ref()
    service = PROVIDERS.catalog_api.create_service(service['id'], service)
    with self.test_client() as c:
        c.delete('/v3/services/%s' % service['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)