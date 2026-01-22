import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_invalid_endpoint_project_association(self):
    """Verify an invalid endpoint-project association is handled."""
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
    endpoint_id2 = uuid.uuid4().hex
    endpoint2 = unit.new_endpoint_ref(service_id=self.service_id, region_id=self.region_id, interface='public', id=endpoint_id2)
    PROVIDERS.catalog_api.create_endpoint(endpoint_id2, endpoint2.copy())
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': endpoint_id2})
    PROVIDERS.catalog_api.delete_endpoint(endpoint_id2)
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.post('/auth/tokens', body=auth_data)
    self.assertValidProjectScopedTokenResponse(r, require_catalog=True, endpoint_filter=True, ep_filter_assoc=1)
    self.assertEqual(self.project['id'], r.result['token']['project']['id'])