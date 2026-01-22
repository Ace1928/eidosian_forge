from unittest import mock
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.resources as resources
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
def test_metadata_show_err_denied_policy(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'metadata', False)
    res_name = 'wibble'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    req = self._get(res_identity._tenant_path() + '/metadata')
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.metadata, req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name)
    self.assertEqual(403, resp.status_int)
    self.assertIn('403 Forbidden', str(resp))