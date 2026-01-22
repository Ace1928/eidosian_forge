from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_clone_from_stacked_branch_preserve_stacking(self):
    try:
        stacked_bzrdir = self.make_stacked_bzrdir()
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    cloned_bzrdir = stacked_bzrdir.clone('cloned', preserve_stacking=True)
    try:
        self.assertEqual(stacked_bzrdir.open_branch().get_stacked_on_url(), cloned_bzrdir.open_branch().get_stacked_on_url())
    except unstackable_format_errors as e:
        pass