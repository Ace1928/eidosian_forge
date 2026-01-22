from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_same_repo(self):
    tree = self.make_branch_and_tree('branch1')
    tree.commit('1st post')
    revid = tree.commit('2st post', allow_pointless=True)
    tree.branch.set_last_revision_info(0, NULL_REVISION)
    tree.branch.import_last_revision_info_and_tags(tree.branch, 2, revid)
    self.assertEqual((2, revid), tree.branch.last_revision_info())