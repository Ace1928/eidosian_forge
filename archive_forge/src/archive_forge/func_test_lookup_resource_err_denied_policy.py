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
def test_lookup_resource_err_denied_policy(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'lookup', False)
    stack_name = 'wibble'
    req = self._get('/stacks/%(stack_name)s/resources' % {'stack_name': stack_name})
    resp = tools.request_with_middleware(fault.FaultWrapper, self.controller.lookup, req, tenant_id=self.tenant, stack_name=stack_name, path='resources')
    self.assertEqual(403, resp.status_int)
    self.assertIn('403 Forbidden', str(resp))