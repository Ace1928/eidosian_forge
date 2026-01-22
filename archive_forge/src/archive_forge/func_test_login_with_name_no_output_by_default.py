from ...tests import TestCaseWithTransport
from . import account
def test_login_with_name_no_output_by_default(self):
    out, err = self.run_bzr(['launchpad-login', '--no-check', 'foo'])
    self.assertEqual('', out)
    self.assertEqual('', err)