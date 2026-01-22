from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_get_graph_stacked(self):
    """A stacked repository shows the graph of its parent."""
    trunk_tree = self.make_branch_and_tree('mainline')
    trunk_revid = trunk_tree.commit('mainline')
    new_branch = self.make_branch('new_branch')
    try:
        new_branch.set_stacked_on_url(trunk_tree.branch.base)
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    new_repo = new_branch.repository
    with new_repo.lock_read():
        self.assertEqual(new_repo.get_parent_map([trunk_revid]), {trunk_revid: (NULL_REVISION,)})