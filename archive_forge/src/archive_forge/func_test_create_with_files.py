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
def test_create_with_files(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'create', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '1')
    template = {u'Foo': u'bar'}
    parameters = {u'InstanceType': u'm1.xlarge'}
    body = {'template': template, 'stack_name': identity.stack_name, 'parameters': parameters, 'files': {'my.yaml': 'This is the file contents.'}, 'timeout_mins': 30}
    req = self._post('/stacks', json.dumps(body))
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=dict(identity))
    result = self.controller.create(req, tenant_id=identity.tenant, body=body)
    expected = {'stack': {'id': '1', 'links': [{'href': self._url(identity), 'rel': 'self'}]}}
    self.assertEqual(expected, result)
    mock_call.assert_called_once_with(req.context, ('create_stack', {'stack_name': identity.stack_name, 'template': template, 'params': {'parameters': parameters, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}, 'files': {'my.yaml': 'This is the file contents.'}, 'environment_files': None, 'files_container': None, 'args': {'timeout_mins': 30}, 'owner_id': None, 'nested_depth': 0, 'user_creds_id': None, 'parent_resource_name': None, 'stack_user_project_id': None, 'template_id': None}), version='1.36')