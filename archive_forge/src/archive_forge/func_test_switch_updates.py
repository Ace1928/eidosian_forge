import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_updates(self):
    """Test switch updates tree and keeps uncommitted changes."""
    checkout, to_branch = self._setup_uncommitted()
    self.assertPathDoesNotExist('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    switch.switch(checkout.controldir, to_branch)
    self.assertPathExists('checkout/file-1')
    self.assertPathDoesNotExist('checkout/file-2')
    self.assertPathExists('checkout/file-3')