import copy
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import extrarouteset
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
from neutronclient.common import exceptions as ncex
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
def test_delete_proper(self):
    t = template_format.parse(template)
    stack = utils.parse_stack(t)
    extra_routes = stack['extrarouteset0']
    scheduler.TaskRunner(extra_routes.create)()
    scheduler.TaskRunner(extra_routes.delete)()
    self.assertEqual((extra_routes.DELETE, extra_routes.COMPLETE), extra_routes.state)
    self.remove_extra_routes_mock.assert_called_once_with('88ce38c4-be8e-11e9-a0a5-5f64570eeec8', {'router': {'routes': [{'destination': '10.0.1.0/24', 'nexthop': '10.0.0.11'}, {'destination': '10.0.2.0/24', 'nexthop': '10.0.0.12'}]}})