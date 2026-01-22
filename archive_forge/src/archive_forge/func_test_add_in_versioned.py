import os
from breezy import osutils, tests
from breezy.tests import features, script
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_add_in_versioned(self):
    """Try to add a file in a versioned directory.

        "brz add" should do this happily.
        """
    tree = self.make_branch_and_tree('.')
    self.build_tree(['inertiatic/', 'inertiatic/esp'])
    self.assertEqual(self.run_bzr('unknowns')[0], 'inertiatic\n')
    self.run_bzr('add --no-recurse inertiatic')
    self.assertEqual(self.run_bzr('unknowns')[0], 'inertiatic/esp\n')
    self.run_bzr('add inertiatic/esp')
    self.assertEqual(self.run_bzr('unknowns')[0], '')