import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_merges_single_merge_rev(self):
    self.assertLogRevnosAndDepths(['-n0', '-r1.1.2'], [('1.1.2', 0), ('1.2.1', 1)], working_dir='level0')