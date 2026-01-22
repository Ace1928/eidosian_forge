import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_auto_one_path(self):
    self.make_abcd_tree()
    out, err = self.run_bzr('mv --auto tree')
    self.assertEqual(out, '')
    self.assertEqual(err, 'a => b\nc => d\n')
    tree = workingtree.WorkingTree.open('tree')
    self.assertTrue(tree.is_versioned('b'))
    self.assertTrue(tree.is_versioned('d'))