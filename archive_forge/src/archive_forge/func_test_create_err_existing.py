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
def test_create_err_existing(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'create', True)
    stack_name = 'wordpress'
    template = {u'Foo': u'bar'}
    parameters = {u'InstanceType': u'm1.xlarge'}
    body = {'template': template, 'stack_name': stack_name, 'parameters': parameters, 'timeout_mins': 30}
    req = self._post('/stacks', json.dumps(body))
    error = heat_exc.StackExists(stack_name='s')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.create, req, tenant_id=self.tenant, body=body)
    self.assertEqual(409, resp.json['code'])
    self.assertEqual('StackExists', resp.json['error']['type'])
    mock_call.assert_called_once_with(req.context, ('create_stack', {'stack_name': stack_name, 'template': template, 'params': {'parameters': parameters, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}, 'files': {}, 'environment_files': None, 'files_container': None, 'args': {'timeout_mins': 30}, 'owner_id': None, 'nested_depth': 0, 'user_creds_id': None, 'parent_resource_name': None, 'stack_user_project_id': None, 'template_id': None}), version='1.36')