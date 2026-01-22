from ... import config
from ...tests import TestCaseInTempDir, TestCaseWithMemoryTransport
from . import account
def test_get_lp_login_unconfigured(self):
    my_config = config.MemoryStack(b'')
    self.assertEqual(None, account.get_lp_login(my_config))