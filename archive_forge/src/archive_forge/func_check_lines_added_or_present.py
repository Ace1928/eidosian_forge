from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def check_lines_added_or_present(self, stacked_branch, revid):
    stacked_repo = stacked_branch.repository
    with stacked_repo.lock_read():
        list(stacked_repo.inventories.iter_lines_added_or_present_in_keys([(revid,)]))