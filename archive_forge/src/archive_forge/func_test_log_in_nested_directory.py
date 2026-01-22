import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_in_nested_directory(self):
    """The log for a directory should show all nested files."""
    self.prepare_tree()
    os.chdir('dir1')
    self.assertLogRevnos(['.'], ['5', '3'])