import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_cannot_get_an_endpoint_group(self):
    endpoint_group = unit.new_endpoint_group_ref(filters={'interface': 'public'})
    endpoint_group = PROVIDERS.catalog_api.create_endpoint_group(endpoint_group['id'], endpoint_group)
    with self.test_client() as c:
        c.get('/v3/OS-EP-FILTER/endpoint_groups/%s' % endpoint_group['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)