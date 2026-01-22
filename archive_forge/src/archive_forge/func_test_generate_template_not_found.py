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
def test_generate_template_not_found(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'generate_template', True)
    req = self._get('/resource_types/NOT_FOUND/template')
    error = heat_exc.EntityNotFound(entity='Resource Type', name='a')
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', side_effect=tools.to_remote_error(error))
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.generate_template, req, tenant_id=self.tenant, type_name='NOT_FOUND')
    self.assertEqual(404, resp.json['code'])
    self.assertEqual('EntityNotFound', resp.json['error']['type'])
    mock_call.assert_called_once_with(req.context, ('generate_template', {'type_name': 'NOT_FOUND', 'template_type': 'cfn'}), version='1.9')