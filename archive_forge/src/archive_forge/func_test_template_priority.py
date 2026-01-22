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
def test_template_priority(self):
    template = {'foo': 'bar', 'blarg': 'wibble'}
    url = 'http://example.com/template'
    body = {'template': template, 'template_url': url}
    data = stacks.InstantiationData(body)
    mock_get = self.patchobject(urlfetch, 'get')
    self.assertEqual(template, data.template())
    mock_get.assert_not_called()