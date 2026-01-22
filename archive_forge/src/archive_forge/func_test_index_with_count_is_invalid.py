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
def test_index_with_count_is_invalid(self, mock_enforce):
    self._mock_enforce_setup(mock_enforce, 'index', True)
    params = {'with_count': 'invalid_value'}
    req = self._get('/stacks', params=params)
    exc = self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, req, tenant_id=self.tenant)
    excepted = 'Unrecognized value "invalid_value" for "with_count", acceptable values are: true, false'
    self.assertIn(excepted, str(exc))