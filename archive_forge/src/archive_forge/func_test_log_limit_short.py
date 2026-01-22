import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_limit_short(self):
    self.make_linear_branch()
    self.assertLogRevnos(['-l', '2'], ['3', '2'])