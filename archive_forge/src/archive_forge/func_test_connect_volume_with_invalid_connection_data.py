from os_brick.initiator.connectors import local
from os_brick.tests.initiator import test_connector
def test_connect_volume_with_invalid_connection_data(self):
    cprops = {}
    self.assertRaises(ValueError, self.connector.connect_volume, cprops)