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
def test_mark_unhealthy_with_invalid_keys(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'mark_unhealthy', True)
    res_name = 'WebServer'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    req = self._get(stack_identity._tenant_path())
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=None)
    body = {'mark_unhealthy': True, rpc_api.RES_STATUS_DATA: 'Any', 'invalid_key1': 1, 'invalid_key2': 2}
    expected = 'Invalid keys in resource mark unhealthy'
    actual = self.assertRaises(webob.exc.HTTPBadRequest, self.controller.mark_unhealthy, req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name, body=body)
    self.assertIn(expected, str(actual))
    self.assertIn('invalid_key1', str(actual))
    self.assertIn('invalid_key2', str(actual))
    mock_call.assert_not_called()