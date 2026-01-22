import uuid
from openstackclient.tests.functional.network.v2 import common
def test_router_set_show_unset(self):
    """Tests create router, set, unset, show"""
    name = uuid.uuid4().hex
    new_name = name + '_'
    cmd_output = self.openstack('router create ' + '--description aaaa ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'router delete ' + new_name)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('aaaa', cmd_output['description'])
    cmd_output = self.openstack('router set ' + '--name ' + new_name + ' --description bbbb ' + '--disable ' + name)
    self.assertOutput('', cmd_output)
    cmd_output = self.openstack('router show ' + new_name, parse_output=True)
    self.assertEqual(new_name, cmd_output['name'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual(False, cmd_output['admin_state_up'])
    self._test_set_router_distributed(new_name)
    cmd_output = self.openstack('router unset ' + '--external-gateway ' + new_name)
    cmd_output = self.openstack('router show ' + new_name, parse_output=True)
    self.assertIsNone(cmd_output['external_gateway_info'])