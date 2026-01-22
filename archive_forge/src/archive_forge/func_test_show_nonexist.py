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
def test_show_nonexist(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'show', True)
    res_name = 'WikiDatabase'
    stack_identity = identifier.HeatIdentifier(self.tenant, 'rubbish', '1')
    res_identity = identifier.ResourceIdentifier(resource_name=res_name, **stack_identity)
    req = self._get(res_identity._tenant_path())
    error = heat_exc.EntityNotFound(entity='Stack', name='a')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.show, req, tenant_id=self.tenant, stack_name=stack_identity.stack_name, stack_id=stack_identity.stack_id, resource_name=res_name)
    self.assertEqual(404, resp.json['code'])
    self.assertEqual('EntityNotFound', resp.json['error']['type'])
    mock_call.assert_called_once_with(req.context, ('describe_stack_resource', {'stack_identity': stack_identity, 'resource_name': res_name, 'with_attr': None}), version='1.2')