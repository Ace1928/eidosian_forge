import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
def test_unshelve_messages_delete(self):
    self.create_tree_with_shelf()
    self.run_script('\n$ cd tree\n$ brz unshelve --delete-only\n2>Deleted changes with id "1".\n')