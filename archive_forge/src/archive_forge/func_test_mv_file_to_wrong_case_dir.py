import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_file_to_wrong_case_dir(self):
    self.requireFeature(CaseInsensitiveFilesystemFeature)
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo/', 'bar'])
    tree.add(['foo', 'bar'])
    out, err = self.run_bzr('mv bar Foo', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: Could not move to Foo: Foo is not versioned.\n', err)