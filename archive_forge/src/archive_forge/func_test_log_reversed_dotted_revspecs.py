import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_reversed_dotted_revspecs(self):
    self.make_merged_branch()
    self.run_bzr_error(('brz: ERROR: Start revision not found in history of end revision.\n',), 'log -r 1.1.1..1')