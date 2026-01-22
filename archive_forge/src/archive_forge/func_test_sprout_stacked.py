from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_stacked(self):
    trunk_tree = self.make_branch_and_tree('mainline')
    trunk_revid = trunk_tree.commit('mainline')
    try:
        new_dir = trunk_tree.controldir.sprout('newbranch', stacked=True)
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    self.assertRevisionNotInRepository('newbranch', trunk_revid)
    tree = new_dir.open_branch().create_checkout('local')
    new_branch_revid = tree.commit('something local')
    self.assertRevisionNotInRepository(trunk_tree.branch.base, new_branch_revid)
    self.assertRevisionInRepository('newbranch', new_branch_revid)