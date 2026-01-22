from os_brick.initiator.connectors import local
from os_brick.tests.initiator import test_connector
def test_get_search_path(self):
    actual = self.connector.get_search_path()
    self.assertIsNone(actual)