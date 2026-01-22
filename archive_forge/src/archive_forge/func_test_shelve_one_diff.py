import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_shelve_one_diff(self):
    tree = self.create_shelvable_tree()
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    shelver = ExpectShelver(tree, tree.basis_tree())
    self.addCleanup(shelver.finalize)
    shelver.expect('Shelve?', 0)
    shelver.expect('Shelve?', 1)
    shelver.expect('Shelve 1 change(s)?', 0)
    shelver.run()
    self.assertFileEqual(LINES_AY, 'tree/foo')