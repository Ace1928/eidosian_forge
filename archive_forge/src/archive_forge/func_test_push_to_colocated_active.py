from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def test_push_to_colocated_active(self):
    tree, rev_1 = self.create_simple_tree()
    dir = self.make_repository('dir').controldir
    try:
        result = dir.push_branch(tree.branch, name='colo')
    except NoColocatedBranchSupport:
        raise TestNotApplicable('no colocated branch support')
    self.assertEqual(tree.branch, result.source_branch)
    self.assertEqual(dir.open_branch(name='colo').base, result.target_branch.base)
    self.assertEqual(dir.open_branch(name='colo').base, tree.branch.get_push_location())