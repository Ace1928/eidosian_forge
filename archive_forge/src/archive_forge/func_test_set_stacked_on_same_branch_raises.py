from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_set_stacked_on_same_branch_raises(self):
    branch = self.make_branch('branch')
    try:
        self.assertRaises(errors.UnstackableLocationError, branch.set_stacked_on_url, '../branch')
    except unstackable_format_errors:
        self.assertRaises(unstackable_format_errors, branch.get_stacked_on_url)
        return
    self.assertRaises(errors.NotStacked, branch.get_stacked_on_url)