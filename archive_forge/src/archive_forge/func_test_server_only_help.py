import os
import fixtures
from openstackclient.tests.functional import base
def test_server_only_help(self):
    """Check list of server-related commands only."""
    raw_output = self.openstack('help server')
    for command in [row[0] for row in self.SERVER_COMMANDS]:
        self.assertIn(command, raw_output)