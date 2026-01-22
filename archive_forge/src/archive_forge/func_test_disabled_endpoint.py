import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_disabled_endpoint(self):
    """Test that a disabled endpoint is handled."""
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': self.endpoint_id})
    disabled_endpoint_ref = copy.copy(self.endpoint)
    disabled_endpoint_id = uuid.uuid4().hex
    disabled_endpoint_ref.update({'id': disabled_endpoint_id, 'enabled': False, 'interface': 'internal'})
    PROVIDERS.catalog_api.create_endpoint(disabled_endpoint_id, disabled_endpoint_ref)
    self.put('/OS-EP-FILTER/projects/%(project_id)s/endpoints/%(endpoint_id)s' % {'project_id': self.project['id'], 'endpoint_id': disabled_endpoint_id})
    auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
    r = self.post('/auth/tokens', body=auth_data)
    endpoints = r.result['token']['catalog'][0]['endpoints']
    endpoint_ids = [ep['id'] for ep in endpoints]
    self.assertEqual([self.endpoint_id], endpoint_ids)