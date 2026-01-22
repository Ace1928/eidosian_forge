from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login_leaves_existing_credentials(self):
    auth = config.AuthenticationConfig()
    auth.set_credentials('Foo', 'bazaar.launchpad.net', 'foo', 'ssh')
    auth.set_credentials('Bar', 'bazaar.staging.launchpad.net', 'foo', 'ssh')
    account._set_global_option('foo')
    account.get_lp_login()
    auth = config.AuthenticationConfig()
    credentials = auth.get_credentials('ssh', 'bazaar.launchpad.net')
    self.assertEqual('Foo', credentials['name'])