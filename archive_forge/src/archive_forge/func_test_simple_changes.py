import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_simple_changes(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add(['file'], ids=[b'file-id'])
    self.assertChangedFileIds([tree.path2id(''), b'file-id'], tree)
    tree.commit('one')
    self.assertChangedFileIds([], tree)