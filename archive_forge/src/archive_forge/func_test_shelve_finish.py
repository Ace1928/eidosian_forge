import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_finish(self):
    tree = self.create_shelvable_tree()
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree())
    self.addCleanup(shelver.finalize)
    shelver.expect('Shelve?', 2)
    shelver.expect('Shelve 2 change(s)?', 0)
    shelver.run()
    self.assertFileEqual(LINES_AJ, 'tree/foo')