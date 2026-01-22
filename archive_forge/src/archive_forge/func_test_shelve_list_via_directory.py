import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_shelve_list_via_directory(self):
    tree = self.make_branch_and_tree('tree')
    creator = self.make_creator(tree)
    shelf_id = tree.get_shelf_manager().shelve_changes(creator, 'Foo')
    out, err = self.run_bzr('shelve -d tree --list', retcode=1)
    self.assertEqual('', err)
    self.assertEqual('  1: Foo\n', out)