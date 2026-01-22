import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_diff_with_merges(self):
    self.assertLogRevnosAndDiff(['-n0'], [('2', 0, self._diff_file2_revno2()), ('1.1.1', 1, self._diff_file2_revno1_1_1()), ('1', 0, self._diff_file1_revno1() + self._diff_file2_revno1())], working_dir='level0')