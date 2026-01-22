import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_files(self):
    """The log for multiple file should only list revs for those files"""
    self.prepare_tree()
    self.assertLogRevnos(['file1', 'file2', 'dir1/dir2/file3'], ['6', '5.1.1', '3', '2', '1'])