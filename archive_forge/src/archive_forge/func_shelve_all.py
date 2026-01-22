import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
@staticmethod
def shelve_all(tree, target_revision_id):
    tree.lock_write()
    try:
        target = tree.branch.repository.revision_tree(target_revision_id)
        shelver = shelf_ui.Shelver(tree, target, auto=True, auto_apply=True)
        try:
            shelver.run()
        finally:
            shelver.finalize()
    finally:
        tree.unlock()