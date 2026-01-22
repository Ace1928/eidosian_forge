import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_restore_uncommitted(self):
    """Test switch updates tree and restores uncommitted changes."""
    checkout, to_branch = self._setup_uncommitted()
    old_branch = self._master_if_present(checkout.branch)
    self.assertPathDoesNotExist('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    self.assertPathExists('checkout/file-3')
    switch.switch(checkout.controldir, to_branch, store_uncommitted=True)
    checkout = workingtree.WorkingTree.open('checkout')
    switch.switch(checkout.controldir, old_branch, store_uncommitted=True)
    self.assertPathDoesNotExist('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    self.assertPathExists('checkout/file-3')