import json
from unittest import mock
from oslo_config import cfg
import webob.exc
import heat.api.middleware.fault as fault
import heat.api.openstack.v1.stacks as stacks
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import exception as heat_exc
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
from heat.tests.api.openstack_v1 import tools
from heat.tests import common
@mock.patch.object(rpc_client.EngineClient, 'call')
def test_update_immutable_parameter(self, mock_call, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'update', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    template = {u'Foo': u'bar'}
    parameters = {u'param1': u'bar'}
    body = {'template': template, 'parameters': parameters, 'files': {}, 'timeout_mins': 30}
    req = self._put('/stacks/%(stack_name)s/%(stack_id)s' % identity, json.dumps(body))
    error = heat_exc.ImmutableParameterModified(keys='param1')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.update, req, tenant_id=identity.tenant, stack_name=identity.stack_name, stack_id=identity.stack_id, body=body)
    self.assertEqual(400, resp.json['code'])
    self.assertEqual('ImmutableParameterModified', resp.json['error']['type'])
    self.assertIn('The following parameters are immutable', str(resp.json['error']['message']))
    mock_call.assert_called_once_with(req.context, ('update_stack', {'stack_identity': dict(identity), 'template': template, 'params': {u'parameters': parameters, u'encrypted_param_names': [], u'parameter_defaults': {}, u'event_sinks': [], u'resource_registry': {}}, 'files': {}, 'environment_files': None, 'files_container': None, 'args': {'timeout_mins': 30}, 'template_id': None}), version='1.36')