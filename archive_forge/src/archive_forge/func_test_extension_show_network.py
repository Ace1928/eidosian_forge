from tempest.lib import exceptions as tempest_exc
from openstackclient.tests.functional import base
def test_extension_show_network(self):
    """Test network extension show"""
    if not self.haz_network:
        self.skipTest('No Network service present')
    name = 'agent'
    output = self.openstack('extension show ' + name, parse_output=True)
    self.assertEqual(name, output.get('alias'))