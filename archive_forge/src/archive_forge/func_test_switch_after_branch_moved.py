import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
def test_switch_after_branch_moved(self):
    """Test switch after the branch is moved."""
    tree = self._setup_tree()
    checkout = tree.branch.create_checkout('checkout', lightweight=self.lightweight)
    self.build_tree(['branch-1/file-2'])
    tree.add('file-2')
    tree.remove('file-1')
    tree.commit('rev2')
    self.build_tree(['checkout/file-3'])
    checkout.add('file-3')
    os.rename('branch-1', 'branch-2')
    to_branch = branch.Branch.open('branch-2')
    err = self.assertRaises((errors.CommandError, errors.NotBranchError), switch.switch, checkout.controldir, to_branch)
    if isinstance(err, errors.CommandError):
        self.assertContainsRe(str(err), 'Unable to connect to current master branch .*To switch anyway, use --force.')
    switch.switch(checkout.controldir, to_branch, force=True)
    self.assertPathDoesNotExist('checkout/file-1')
    self.assertPathExists('checkout/file-2')
    self.assertPathExists('checkout/file-3')