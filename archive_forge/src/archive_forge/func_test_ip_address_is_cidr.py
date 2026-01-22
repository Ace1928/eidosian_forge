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
def test_ip_address_is_cidr(self):
    t = template_format.parse(neutron_port_with_address_pair_template)
    t['resources']['port']['properties']['allowed_address_pairs'][0]['ip_address'] = '10.0.3.0/24'
    stack = utils.parse_stack(t)
    self.find_mock.return_value = 'abcd1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': '2e00180a-ff9d-42c4-b701-a0606b243447'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': '2e00180a-ff9d-42c4-b701-a0606b243447'}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.create_mock.assert_called_once_with({'port': {'network_id': u'abcd1234', 'allowed_address_pairs': [{'ip_address': u'10.0.3.0/24', 'mac_address': u'00-B0-D0-86-BB-F7'}], 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'binding:vnic_type': 'normal', 'device_owner': '', 'device_id': ''}})