import json
import uuid
import http.client
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_list_head_policies(self):
    """Call ``GET & HEAD /policies``."""
    resource_url = '/policies'
    r = self.get(resource_url)
    self.assertValidPolicyListResponse(r, ref=self.policy)
    self.head(resource_url, expected_status=http.client.OK)