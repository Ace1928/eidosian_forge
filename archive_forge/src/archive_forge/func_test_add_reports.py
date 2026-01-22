import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_reports(self):
    """add command prints the names of added files."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['top.txt', 'dir/', 'dir/sub.txt', 'CVS'])
    self.build_tree_contents([('.bzrignore', b'CVS\n')])
    out = self.run_bzr('add')[0]
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual(['adding .bzrignore', 'adding dir', 'adding dir/sub.txt', 'adding top.txt'], results)
    out = self.run_bzr('add -v')[0]
    results = sorted(out.rstrip('\n').split('\n'))
    self.assertEqual(['ignored CVS matching "CVS"'], results)