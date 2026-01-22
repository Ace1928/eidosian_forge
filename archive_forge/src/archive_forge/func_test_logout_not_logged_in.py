from ...tests import TestCaseWithTransport
from . import account
def test_logout_not_logged_in(self):
    out, err = self.run_bzr(['launchpad-logout', '-v'], retcode=1)
    self.assertEqual('Not logged into Launchpad.\n', out)
    self.assertEqual('', err)