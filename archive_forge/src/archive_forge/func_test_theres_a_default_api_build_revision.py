from unittest import mock
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.build_info as build_info
from heat.common import policy
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_theres_a_default_api_build_revision(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'build_info', True)
    req = self._get('/build_info')
    self.controller.rpc_client = mock.Mock()
    response = self.controller.build_info(req, tenant_id=self.tenant)
    self.assertIn('api', response)
    self.assertIn('revision', response['api'])
    self.assertEqual('unknown', response['api']['revision'])