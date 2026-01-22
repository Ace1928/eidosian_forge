from ...tests import TestCaseWithTransport
from . import account
def test_login_with_name_sets_login(self):
    self.run_bzr(['launchpad-login', '--no-check', 'foo'])
    self.assertEqual('foo', account.get_lp_login())