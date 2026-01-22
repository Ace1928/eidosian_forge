import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_creation(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('add tree root')
    self.build_tree(['tree/foo'])
    tree.add('foo')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree(), reporter=shelf_ui.ApplyReporter())
    self.addCleanup(shelver.finalize)
    shelver.expect('Delete file "foo"?', 0)
    shelver.expect('Apply 1 change(s)?', 0)
    shelver.run()
    self.assertPathDoesNotExist('tree/foo')