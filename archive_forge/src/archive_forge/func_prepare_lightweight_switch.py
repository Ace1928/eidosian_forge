import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def prepare_lightweight_switch(self):
    branch = self.make_branch('branch')
    branch.create_checkout('tree', lightweight=True)
    osutils.rename('branch', 'branch1')