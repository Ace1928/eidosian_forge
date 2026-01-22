import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_null_begin_revspec(self):
    self.make_linear_branch()
    self.assertLogRevnos(['-r..3'], ['3', '2', '1'])