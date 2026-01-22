import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_scoped_token_with_no_catalog_using_endpoint_filter(self):
    """Verify endpoint filter does not affect no catalog."""
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.post('/auth/tokens?nocatalog', body=auth_data)
    self.assertValidProjectScopedTokenResponse(r, require_catalog=False)
    self.assertEqual(self.project['id'], r.result['token']['project']['id'])