import uuid
from openstackclient.tests.functional.network.v2 import common
def test_ip_availability_show(self):
    """Test ip availability show"""
    cmd_output = self.openstack('ip availability show ' + self.NETWORK_NAME, parse_output=True)
    self.assertEqual(self.NETWORK_NAME, cmd_output['network_name'])