import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_dash_d(self):
    self.example_branch('a')
    self.make_branch_and_tree('b')
    self.make_branch_and_tree('c')
    self.run_bzr('pull -d b a')
    c_url = urlutils.local_path_to_url('c')
    self.assertStartsWith(c_url, 'file://')
    self.run_bzr(['pull', '-d', c_url, 'a'])