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
def test_update_with_tags(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'update', True)
    identity = identifier.HeatIdentifier(self.tenant, 'wordpress', '6')
    template = {u'Foo': u'bar'}
    parameters = {u'InstanceType': u'm1.xlarge'}
    body = {'template': template, 'parameters': parameters, 'files': {}, 'tags': 'tag1,tag2', 'timeout_mins': 30}
    req = self._put('/stacks/%(stack_name)s/%(stack_id)s' % identity, json.dumps(body))
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=dict(identity))
    self.assertRaises(webob.exc.HTTPAccepted, self.controller.update, req, tenant_id=identity.tenant, stack_name=identity.stack_name, stack_id=identity.stack_id, body=body)
    mock_call.assert_called_once_with(req.context, ('update_stack', {'stack_identity': dict(identity), 'template': template, 'params': {'parameters': parameters, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {}}, 'files': {}, 'environment_files': None, 'files_container': None, 'args': {'timeout_mins': 30, 'tags': ['tag1', 'tag2']}, 'template_id': None}), version='1.36')