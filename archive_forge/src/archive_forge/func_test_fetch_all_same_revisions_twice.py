from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_all_same_revisions_twice(self):
    repo = self.make_repository('repo')
    tree = self.make_branch_and_tree('tree')
    revision_id = tree.commit('test')
    repo.fetch(tree.branch.repository)
    repo.fetch(tree.branch.repository)