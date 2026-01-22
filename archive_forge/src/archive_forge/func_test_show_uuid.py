from ironicclient.tests.functional.osc.v1 import base
def test_show_uuid(self):
    """Check baremetal port show command with UUID.

        Test steps:
        1) Create baremetal port in setUp.
        2) Show baremetal port calling it by UUID.
        3) Check port fields in output.
        """
    port = self.port_show(self.port['uuid'])
    self.assertEqual(self.port['address'], port['address'])
    self.assertEqual(self.port['uuid'], port['uuid'])
    self.assertEqual(self.port['node_uuid'], self.node['uuid'])