import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_no_shelved_changes(self):
    tree = self.make_branch_and_tree('.')
    err = self.run_bzr('shelve --list')[1]
    self.assertEqual('No shelved changes.\n', err)