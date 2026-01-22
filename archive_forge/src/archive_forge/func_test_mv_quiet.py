import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_quiet(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['aaa'])
    tree.add(['aaa'])
    out, err = self.run_bzr('mv --quiet aaa bbb')
    self.assertEqual(out, '')
    self.assertEqual(err, '')