import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_default_scoped_token_using_endpoint_filter(self):
    """Verify endpoints from default scoped token filtered."""
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.post('/auth/tokens', body=auth_data)
    self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
    self.assertEqual(self.project['id'], r.result['token']['project']['id'])
    self.assertIn('name', r.result['token']['catalog'][0])
    endpoint = r.result['token']['catalog'][0]['endpoints'][0]
    self.assertIn('region', endpoint)
    self.assertIn('region_id', endpoint)
    self.assertEqual(endpoint['region'], endpoint['region_id'])