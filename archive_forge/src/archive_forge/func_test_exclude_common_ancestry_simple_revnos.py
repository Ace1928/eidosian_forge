import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_exclude_common_ancestry_simple_revnos(self):
    self.make_linear_branch()
    self.assertLogRevnos(['-r1..3', '--exclude-common-ancestry'], ['3', '2'])