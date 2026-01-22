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
def test_resource_schema_faulty_template(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'resource_schema', True)
    req = self._get('/resource_types/FaultyTemplate')
    type_name = 'FaultyTemplate'
    error = heat_exc.InvalidGlobalResource(type_name='FaultyTemplate')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.resource_schema, req, tenant_id=self.tenant, type_name=type_name)
    self.assertEqual(500, resp.json['code'])
    self.assertEqual('InvalidGlobalResource', resp.json['error']['type'])
    mock_call.assert_called_once_with(req.context, ('resource_schema', {'type_name': type_name, 'with_description': False}), version='1.30')