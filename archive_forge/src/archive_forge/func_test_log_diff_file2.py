import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_diff_file2(self):
    self.assertLogRevnosAndDiff(['-n1', 'file2'], [('2', 0, self._diff_file2_revno2()), ('1', 0, self._diff_file2_revno1())], working_dir='level0')