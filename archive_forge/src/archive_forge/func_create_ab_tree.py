import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def create_ab_tree(self):
    """Create a committed test tree with two files"""
    source = self.make_branch_and_tree('source')
    self.build_tree_contents([('source/file1', b'A')])
    self.build_tree_contents([('source/file2', b'B')])
    source.add(['file1', 'file2'], ids=[b'file1-id', b'file2-id'])
    source.commit('commit files')
    source.lock_write()
    self.addCleanup(source.unlock)
    return source