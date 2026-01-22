import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_diff_across_rename(self):
    """The working tree path should always be considered for diffing"""
    tree = self.make_example_branch()
    self.run_bzr('diff -r 0..1 hello', retcode=1)
    tree.rename_one('hello', 'hello1')
    self.run_bzr('diff hello1', retcode=1)
    self.run_bzr('diff -r 0..1 hello1', retcode=1)