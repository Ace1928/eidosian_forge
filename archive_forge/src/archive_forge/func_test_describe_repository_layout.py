import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_repository_layout(self):
    repository = self.make_repository('.', shared=True)
    tree = controldir.ControlDir.create_branch_convenience('tree', force_new_tree=True).controldir.open_workingtree()
    self.assertEqual('Shared repository with trees and colocated branches', info.describe_layout(tree.branch.repository, control=tree.controldir))
    repository.set_make_working_trees(False)
    self.assertEqual('Shared repository with colocated branches', info.describe_layout(tree.branch.repository, control=tree.controldir))
    self.assertEqual('Repository branch', info.describe_layout(tree.branch.repository, tree.branch, control=tree.controldir))
    self.assertEqual('Repository branchless tree', info.describe_layout(tree.branch.repository, None, tree, control=tree.controldir))
    self.assertEqual('Repository tree', info.describe_layout(tree.branch.repository, tree.branch, tree, control=tree.controldir))
    tree.branch.bind(tree.branch)
    self.assertEqual('Repository checkout', info.describe_layout(tree.branch.repository, tree.branch, tree, control=tree.controldir))
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    self.assertEqual('Lightweight checkout', info.describe_layout(checkout.branch.repository, checkout.branch, checkout, control=tree.controldir))