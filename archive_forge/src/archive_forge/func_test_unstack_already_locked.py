from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_unstack_already_locked(self):
    """Removing the stacked-on branch with an already write-locked branch
        works.

        This was bug 551525.
        """
    try:
        stacked_bzrdir = self.make_stacked_bzrdir()
    except unstackable_format_errors as e:
        raise TestNotApplicable(e)
    stacked_branch = stacked_bzrdir.open_branch()
    stacked_branch.lock_write()
    stacked_branch.set_stacked_on_url(None)
    stacked_branch.unlock()