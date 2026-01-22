from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def make_two_commits(self, change_root, fetch_twice):
    self.make_tree_and_repo()
    self.tree.commit('first commit', rev_id=b'first-id')
    if change_root:
        self.tree.set_root_id(b'unique-id')
    self.tree.commit('second commit', rev_id=b'second-id')
    if fetch_twice:
        self.repo.fetch(self.tree.branch.repository, b'first-id')
    self.repo.fetch(self.tree.branch.repository, b'second-id')