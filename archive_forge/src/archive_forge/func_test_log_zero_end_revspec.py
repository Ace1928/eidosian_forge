import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_zero_end_revspec(self):
    self.make_linear_branch()
    self.run_bzr_error(['brz: ERROR: Logging revision 0 is invalid.'], ['log', '-r-2..0'])