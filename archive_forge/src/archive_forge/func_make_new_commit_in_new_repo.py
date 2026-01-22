import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def make_new_commit_in_new_repo(self, trunk_repo, parents=None):
    tree = self.branch_trunk_and_make_tree(trunk_repo, 'branch')
    tree.set_parent_ids(parents)
    tree.commit('Branch commit', rev_id=b'rev-2')
    branch_repo = tree.branch.repository
    branch_repo.lock_read()
    self.addCleanup(branch_repo.unlock)
    return branch_repo