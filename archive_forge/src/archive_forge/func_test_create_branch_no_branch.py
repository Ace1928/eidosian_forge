import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_create_branch_no_branch(self):
    self.prepare_lightweight_switch()
    self.run_bzr_error(['cannot create branch without source branch'], 'switch --create-branch ../branch2', working_dir='tree')