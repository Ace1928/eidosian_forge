import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_restore_uncommitted_same_revision(self):
    """Test switch updates tree and restores uncommitted changes."""
    checkout, to_branch = self._setup_uncommitted(same_revision=True)
    old_branch = self._master_if_present(checkout.branch)
    switch.switch(checkout.controldir, to_branch, store_uncommitted=True)
    checkout = workingtree.WorkingTree.open('checkout')
    switch.switch(checkout.controldir, old_branch, store_uncommitted=True)
    self.assertPathExists('checkout/file-3')