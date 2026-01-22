from ironicclient.tests.functional.osc.v1 import base
def test_on_reboot_on(self):
    """Reboot node from Power ON state.

        Test steps:
        1) Create baremetal node in setUp.
        2) Set node Power State ON as precondition.
        3) Call reboot command for baremetal node.
        4) Check node Power State ON in node properties.
        """
    self.openstack('baremetal node power on {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['power_state'])
    self.assertEqual('power on', show_prop['power_state'])
    self.openstack('baremetal node reboot {0}'.format(self.node['uuid']))
    show_prop = self.node_show(self.node['uuid'], ['power_state'])
    self.assertEqual('power on', show_prop['power_state'])