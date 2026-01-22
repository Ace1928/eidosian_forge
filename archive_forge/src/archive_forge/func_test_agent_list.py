from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_agent_list(self):
    ex = self.assertRaises(exceptions.CommandFailed, self.nova, 'agent-list')
    self.assertIn('This resource is no longer available. No forwarding address is given. (HTTP 410)', str(ex))
    self.assertIn('This command has been deprecated since 23.0.0 Wallaby Release and will be removed in the first major release after the Nova server 24.0.0 X release.', str(ex.stderr))
    ex = self.assertRaises(exceptions.CommandFailed, self.nova, 'agent-list', flags='--debug')
    self.assertIn('This resource is no longer available. No forwarding address is given. (HTTP 410)', str(ex))
    self.assertIn('This command has been deprecated since 23.0.0 Wallaby Release and will be removed in the first major release after the Nova server 24.0.0 X release.', str(ex.stderr))