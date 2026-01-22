from ...tests import TestCaseWithTransport
from . import account
def test_logout(self):
    out, err = self.run_bzr(['launchpad-login', '-v', '--no-check', 'foo'])
    self.assertEqual("Launchpad user ID set to 'foo'.\n", out)
    self.assertEqual('', err)
    out, err = self.run_bzr(['launchpad-logout', '-v'])
    self.assertEqual('Launchpad user ID foo logged out.\n', out)
    self.assertEqual('', err)