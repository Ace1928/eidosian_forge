import os
import fixtures
from openstackclient.tests.functional import base
def test_server_commands_main_help(self):
    """Check server commands in main help message."""
    raw_output = self.openstack('help')
    for command, description in self.SERVER_COMMANDS:
        msg = 'Command: %s not found in help output:\n%s' % (command, raw_output)
        self.assertIn(command, raw_output, msg)
        msg = 'Description: %s not found in help output:\n%s' % (description, raw_output)
        self.assertIn(description, raw_output, msg)