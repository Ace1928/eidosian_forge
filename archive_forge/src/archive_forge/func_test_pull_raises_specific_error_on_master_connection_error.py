from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_pull_raises_specific_error_on_master_connection_error(self):
    master_tree = self.make_from_branch_and_tree('master')
    checkout = master_tree.branch.create_checkout('checkout')
    other = self.sprout_to(master_tree.branch.controldir, 'other').open_branch()
    try:
        master_tree.branch.controldir.destroy_branch()
    except errors.UnsupportedOperation:
        raise TestNotApplicable('control format does not support destroying default branch')
    self.assertRaises(errors.BoundBranchConnectionFailure, checkout.branch.pull, other)