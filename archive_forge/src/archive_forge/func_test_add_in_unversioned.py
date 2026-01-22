import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_in_unversioned(self):
    """Try to add a file in an unversioned directory.

        "brz add" should add the parent(s) as necessary.
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['inertiatic/', 'inertiatic/esp'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'inertiatic\n')
    self.run_bzr('add inertiatic/esp')
    self.assertEqual(self.run_bzr('unknowns')[0], '')
    self.build_tree(['veil/', 'veil/cerpin/', 'veil/cerpin/taxt'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'veil\n')
    self.run_bzr('add veil/cerpin/taxt')
    self.assertEqual(self.run_bzr('unknowns')[0], '')
    self.build_tree(['cicatriz/', 'cicatriz/esp'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'cicatriz\n')
    self.run_bzr('add inertiatic/../cicatriz/esp')
    self.assertEqual(self.run_bzr('unknowns')[0], '')