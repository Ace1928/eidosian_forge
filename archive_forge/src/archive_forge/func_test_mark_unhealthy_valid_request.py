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
def test_mark_unhealthy_valid_request(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'mark_unhealthy', True)
    res_name = 'WebServer'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    req = self._get(stack_identity._tenant_path())
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=None)
    body = {'mark_unhealthy': True, rpc_api.RES_STATUS_DATA: 'Anything'}
    params = {'stack_identity': stack_identity, 'resource_name': res_name}
    params.update(body)
    result = self.controller.mark_unhealthy(req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name, body=body)
    self.assertIsNone(result)
    mock_call.assert_called_once_with(req.context, ('resource_mark_unhealthy', params), version='1.26')