import os
import fixtures
from openstackclient.tests.functional import base
def test_commands_help_no_auth(self):
    """Check help commands without auth info."""
    for key in os.environ.keys():
        if key.startswith('OS_'):
            self.useFixture(fixtures.EnvironmentVariable(key, None))
    raw_output = self.openstack('help')
    self.assertIn('usage: openstack', raw_output)
    raw_output = self.openstack('--help')
    self.assertIn('usage: openstack', raw_output)
    raw_output = self.openstack('help network list')
    self.assertIn('List networks', raw_output)
    raw_output = self.openstack('network list --help')
    self.assertIn('List networks', raw_output)
    raw_output = self.openstack('help volume list')
    self.assertIn('List volumes', raw_output)
    raw_output = self.openstack('volume list --help')
    self.assertIn('List volumes', raw_output)
    raw_output = self.openstack('help server list')
    self.assertIn('List servers', raw_output)
    raw_output = self.openstack('server list --help')
    self.assertIn('List servers', raw_output)
    raw_output = self.openstack('help user list')
    self.assertIn('List users', raw_output)
    raw_output = self.openstack('user list --help')
    self.assertIn('List users', raw_output)
    raw_output = self.openstack('help image list')
    self.assertIn('List available images', raw_output)
    raw_output = self.openstack('image list --help')
    self.assertIn('List available images', raw_output)
    raw_output = self.openstack('help container list')
    self.assertIn('List containers', raw_output)
    raw_output = self.openstack('container list --help')
    self.assertIn('List containers', raw_output)