import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_nonexistent_file(self):
    self.make_minimal_branch()
    out, err = self.run_bzr('log does-not-exist', retcode=3)
    self.assertContainsRe(err, 'Path unknown at end or start of revision range: does-not-exist')