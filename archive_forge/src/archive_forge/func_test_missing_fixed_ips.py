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
def test_missing_fixed_ips(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties'].pop('fixed_ips')
    stack = utils.parse_stack(t)
    self.find_mock.return_value = 'net1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', 'fixed_ips': {'subnet_id': 'd0e971a6-a6b4-4f4c-8c88-b75e9c120b7e', 'ip_address': '10.0.0.2'}}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.create_mock.assert_called_once_with({'port': {'network_id': u'net1234', 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'binding:vnic_type': 'normal', 'device_id': ''}})