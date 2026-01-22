import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_gather_location_shared_repo_checkout(self):
    tree = self.make_branch_and_tree('tree')
    srepo = self.make_repository('shared', shared=True)
    shared_checkout = tree.branch.create_checkout('shared/checkout')
    self.assertEqual([('repository checkout root', shared_checkout.controldir.root_transport.base), ('checkout of branch', tree.controldir.root_transport.base), ('shared repository', srepo.controldir.root_transport.base)], self.gather_tree_location_info(shared_checkout))