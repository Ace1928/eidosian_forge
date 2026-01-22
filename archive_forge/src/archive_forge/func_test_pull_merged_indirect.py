from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_pull_merged_indirect(self):
    parent = self.make_from_branch_and_tree('parent')
    parent.commit('1st post', allow_pointless=True)
    try:
        mine = self.sprout_to(parent.controldir, 'mine').open_workingtree()
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    mine.commit('my change', allow_pointless=True)
    other = self.sprout_to(parent.controldir, 'other').open_workingtree()
    other.merge_from_branch(mine.branch)
    other.commit('merge my change')
    try:
        parent.merge_from_branch(other.branch)
    except errors.NoRoundtrippingSupport:
        raise TestNotApplicable('lossless push between %r and %r not supported' % (self.branch_format_from, self.branch_format_to))
    p2 = parent.commit('merge other')
    mine.pull(parent.branch)
    self.assertEqual(p2, mine.branch.last_revision())