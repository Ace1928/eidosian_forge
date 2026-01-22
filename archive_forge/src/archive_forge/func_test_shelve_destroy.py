import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_shelve_destroy(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['file'])
    tree.add('file')
    self.run_bzr('shelve --all --destroy')
    self.assertPathDoesNotExist('file')
    self.assertIs(None, tree.get_shelf_manager().last_shelf())