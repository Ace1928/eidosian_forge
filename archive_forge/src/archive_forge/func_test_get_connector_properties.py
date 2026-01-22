from os_brick.initiator.connectors import local
from os_brick.tests.initiator import test_connector
def test_get_connector_properties(self):
    props = local.LocalConnector.get_connector_properties('sudo', multipath=True, enforce_multipath=True)
    expected_props = {}
    self.assertEqual(expected_props, props)