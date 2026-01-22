from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login_errors_on_mismatch(self):
    account._set_auth_user('foo')
    account._set_global_option('bar')
    e = self.assertRaises(account.MismatchedUsernames, account.get_lp_login)
    self.assertEqual('breezy.conf and authentication.conf disagree about launchpad account name.  Please re-run launchpad-login.', str(e))