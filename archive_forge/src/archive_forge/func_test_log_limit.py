import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_limit(self):
    tree = self.make_branch_and_tree('.')
    for pos in range(10):
        tree.commit('%s' % pos)
    self.assertLogRevnos(['--limit', '2'], ['10', '9'])