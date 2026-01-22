import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_auto_dry_run(self):
    self.make_abcd_tree()
    out, err = self.run_bzr('mv --auto --dry-run', working_dir='tree')
    self.assertEqual(out, '')
    self.assertEqual(err, 'a => b\nc => d\n')
    tree = workingtree.WorkingTree.open('tree')
    self.assertTrue(tree.is_versioned('a'))
    self.assertTrue(tree.is_versioned('c'))