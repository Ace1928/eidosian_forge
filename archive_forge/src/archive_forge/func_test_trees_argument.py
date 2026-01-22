import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_trees_argument(self):
    self.run_bzr('init-shared-repo --trees trees')
    repo = ControlDir.open('trees').open_repository()
    self.assertEqual(True, repo.make_working_trees())