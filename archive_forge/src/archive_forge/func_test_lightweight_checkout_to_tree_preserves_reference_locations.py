from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_lightweight_checkout_to_tree_preserves_reference_locations(self):
    format = controldir.format_registry.make_controldir('1.9')
    format.set_branch_format(_mod_bzrbranch.BzrBranchFormat8())
    tree = self.make_branch_and_tree('tree', format=format)
    tree.branch.set_reference_info(b'file_id', '../location', 'path')
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    reconfiguration = reconfigure.Reconfigure.to_tree(checkout.controldir)
    reconfiguration.apply()
    checkout_branch = checkout.controldir.open_branch()
    self.assertEqual(('../location', 'path'), checkout_branch.get_reference_info(b'file_id'))