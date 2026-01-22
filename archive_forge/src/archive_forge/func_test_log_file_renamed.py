import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_file_renamed(self):
    """File matched against revision range, not current tree."""
    self.prepare_tree(complex=True)
    err_msg = 'Path unknown at end or start of revision range: file3'
    err = self.run_bzr('log file3', retcode=3)[1]
    self.assertContainsRe(err, err_msg)
    self.assertLogRevnos(['-r..4', 'file3'], ['3'])