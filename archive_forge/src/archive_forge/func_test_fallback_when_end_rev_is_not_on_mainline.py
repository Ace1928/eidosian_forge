import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_fallback_when_end_rev_is_not_on_mainline(self):
    self.assertLogRevnos(['-n1', '-r1.1.1..5.1.1'], ['5.1.1', '5', '4', '3'])