import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_subdir_add(self):
    """Add in subdirectory should add only things from there down"""
    eq = self.assertEqual
    ass = self.assertTrue
    t = self.make_branch_and_tree('.')
    b = t.branch
    self.build_tree(['src/', 'README'])
    eq(sorted(t.unknowns()), ['README', 'src'])
    self.run_bzr('add src')
    self.build_tree(['src/foo.c'])
    self.run_bzr('add', working_dir='src')
    self.assertEqual('README\n', self.run_bzr('unknowns', working_dir='src')[0])
    t = t.controldir.open_workingtree('src')
    versioned = [path for path, entry in t.iter_entries_by_dir()]
    self.assertEqual(versioned, ['', 'src', 'src/foo.c'])
    self.run_bzr('add')
    self.assertEqual(self.run_bzr('unknowns')[0], '')
    self.run_bzr('check')