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
def test_port_with_dns_name(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties']['dns_name'] = 'myvm'
    stack = utils.parse_stack(t)
    port_prop = {'network_id': u'net_or_sub', 'dns_name': 'myvm', 'fixed_ips': [{'subnet_id': u'net_or_sub', 'ip_address': u'10.0.3.21'}], 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'binding:vnic_type': 'normal', 'device_id': ''}
    self._mock_create_with_props()
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.assertEqual('my-vm.openstack.org.', port.FnGetAtt('dns_assignment')['fqdn'])
    self.assertEqual((port.CREATE, port.COMPLETE), port.state)
    self.create_mock.assert_called_once_with({'port': port_prop})