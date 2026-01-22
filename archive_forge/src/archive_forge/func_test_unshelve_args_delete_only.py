import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve_args_delete_only(self):
    tree = self.make_branch_and_tree('tree')
    manager = tree.get_shelf_manager()
    shelf_file = manager.new_shelf()[1]
    try:
        shelf_file.write(b'garbage')
    finally:
        shelf_file.close()
    unshelver = shelf_ui.Unshelver.from_args(directory='tree', action='delete-only')
    try:
        unshelver.run()
    finally:
        unshelver.tree.unlock()
    self.assertIs(None, manager.last_shelf())