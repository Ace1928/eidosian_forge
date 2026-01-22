from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def prepare_stacked_on_fetch(self):
    stack_on = self.make_branch_and_tree('stack-on')
    rev1 = stack_on.commit('first commit')
    try:
        stacked_dir = stack_on.controldir.sprout('stacked', stacked=True)
    except unstackable_format_errors as e:
        raise TestNotApplicable('Format does not support stacking.')
    unstacked = self.make_repository('unstacked')
    return (stacked_dir.open_workingtree(), unstacked, rev1)