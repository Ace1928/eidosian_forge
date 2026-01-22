from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_failed_port_no_replace(self):
    t = template_format.parse(neutron_port_template)
    stack = utils.parse_stack(t)
    port = stack['port']
    port.resource_id = 'r_id'
    port.state_set(port.CREATE, port.FAILED)
    new_props = port.properties.data.copy()
    new_props['name'] = 'new_one'
    self.find_mock.return_value = 'net_or_sub'
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'fixed_ips': {'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.3.21'}}}
    update_snippet = rsrc_defn.ResourceDefinition(port.name, port.type(), new_props)
    scheduler.TaskRunner(port.update, update_snippet)()
    self.assertEqual((port.UPDATE, port.COMPLETE), port.state)
    self.assertEqual(1, self.update_mock.call_count)