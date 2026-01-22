from unittest import mock
from blazarclient import exception as client_exception
from oslo_utils.fixture import uuidsentinel as uuids
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import blazar
from heat.engine.resources.openstack.blazar import host
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_parse_extra_capability(self):
    t = template_format.parse(blazar_host_template_extra_capability)
    stack = utils.parse_stack(t)
    resource_defns = self.stack.t.resource_definitions(stack)
    rsrc_defn = resource_defns['test-host']
    host_resource = self._create_resource('host', rsrc_defn, stack)
    args = dict(((k, v) for k, v in host_resource.properties.items() if v is not None))
    parsed_args = host_resource._parse_extra_capability(args)
    self.assertEqual({'gpu': True, 'name': 'test-host'}, parsed_args)