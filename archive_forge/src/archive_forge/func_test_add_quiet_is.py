import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_quiet_is(self):
    """add -q does not print the names of added files."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['top.txt', 'dir/', 'dir/sub.txt'])
    out = self.run_bzr('add -q')[0]
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual([''], results)