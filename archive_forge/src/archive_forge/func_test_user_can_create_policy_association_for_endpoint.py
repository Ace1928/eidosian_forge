import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def test_user_can_create_policy_association_for_endpoint(self):
    policy = unit.new_policy_ref()
    policy = PROVIDERS.policy_api.create_policy(policy['id'], policy)
    service = PROVIDERS.catalog_api.create_service(uuid.uuid4().hex, unit.new_service_ref())
    endpoint = unit.new_endpoint_ref(service['id'], region_id=None)
    endpoint = PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
    with self.test_client() as c:
        c.put('/v3/policies/%s/OS-ENDPOINT-POLICY/endpoints/%s' % (policy['id'], endpoint['id']), headers=self.headers, expected_status_code=http.client.NO_CONTENT)