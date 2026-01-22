import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_quit(self):
    tree = self.create_shelvable_tree()
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree())
    self.addCleanup(shelver.finalize)
    shelver.expect('Shelve?', 3)
    self.assertRaises(errors.UserAbort, shelver.run)
    self.assertFileEqual(LINES_ZY, 'tree/foo')