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
def test_list_template_versions(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'list_template_versions', True)
    req = self._get('/template_versions')
    engine_response = [{'version': 'heat_template_version.2013-05-23', 'type': 'hot'}, {'version': 'AWSTemplateFormatVersion.2010-09-09', 'type': 'cfn'}]
    mock_call = self.patchobject(rpc_client.EngineClient, 'call', return_value=engine_response)
    response = self.controller.list_template_versions(req, tenant_id=self.tenant)
    self.assertEqual({'template_versions': engine_response}, response)
    mock_call.assert_called_once_with(req.context, ('list_template_versions', {}), version='1.11')