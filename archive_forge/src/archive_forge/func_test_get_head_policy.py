import json
import uuid
import http.client
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_policy(self):
    """Call ``GET & HEAD /policies/{policy_id}``."""
    resource_url = '/policies/%(policy_id)s' % {'policy_id': self.policy_id}
    r = self.get(resource_url)
    self.assertValidPolicyResponse(r, self.policy)
    self.head(resource_url, expected_status=http.client.OK)