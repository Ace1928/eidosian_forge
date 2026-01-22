import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_policy_for_endpoint(self):
    """GET /endpoints/{endpoint_id}/policy."""
    self.put('/policies/%(policy_id)s/OS-ENDPOINT-POLICY/endpoints/%(endpoint_id)s' % {'policy_id': self.policy['id'], 'endpoint_id': self.endpoint['id']})
    self.head('/endpoints/%(endpoint_id)s/OS-ENDPOINT-POLICY/policy' % {'endpoint_id': self.endpoint['id']}, expected_status=http.client.OK)
    r = self.get('/endpoints/%(endpoint_id)s/OS-ENDPOINT-POLICY/policy' % {'endpoint_id': self.endpoint['id']})
    self.assertValidPolicyResponse(r, ref=self.policy)