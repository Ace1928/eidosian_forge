import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_subnet_set_show_unset(self):
    """Test create subnet, set, unset, show"""
    name = uuid.uuid4().hex
    new_name = name + '_'
    cmd = 'subnet create ' + '--network ' + self.NETWORK_NAME + ' --description aaaa --subnet-range'
    cmd_output = self._subnet_create(cmd, name)
    self.addCleanup(self.openstack, 'subnet delete ' + new_name)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('aaaa', cmd_output['description'])
    cmd_output = self.openstack('subnet set ' + '--name ' + new_name + ' --description bbbb ' + '--no-dhcp ' + '--gateway 10.10.11.1 ' + name)
    self.assertOutput('', cmd_output)
    cmd_output = self.openstack('subnet show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual(False, cmd_output['enable_dhcp'])
    self.assertEqual('10.10.11.1', cmd_output['gateway_ip'])
    cmd_output = self.openstack('subnet unset --gateway ' + new_name)
    self.assertOutput('', cmd_output)
    cmd_output = self.openstack('subnet show ' + new_name, parse_output=True)
    self.assertIsNone(cmd_output['gateway_ip'])