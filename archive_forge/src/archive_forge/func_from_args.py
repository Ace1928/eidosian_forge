import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
@classmethod
def from_args(klass, shelf_id=None, action='apply', directory='.', write_diff_to=None):
    """Create an unshelver from commandline arguments.

        The returned shelver will have a tree that is locked and should
        be unlocked.

        :param shelf_id: Integer id of the shelf, as a string.
        :param action: action to perform.  May be 'apply', 'dry-run',
            'delete', 'preview'.
        :param directory: The directory to unshelve changes into.
        :param write_diff_to: See Unshelver.__init__().
        """
    tree, path = workingtree.WorkingTree.open_containing(directory)
    tree.lock_tree_write()
    try:
        manager = tree.get_shelf_manager()
        if shelf_id is not None:
            try:
                shelf_id = int(shelf_id)
            except ValueError:
                raise shelf.InvalidShelfId(shelf_id)
        else:
            shelf_id = manager.last_shelf()
            if shelf_id is None:
                raise errors.CommandError(gettext('No changes are shelved.'))
        apply_changes = True
        delete_shelf = True
        read_shelf = True
        show_diff = False
        if action == 'dry-run':
            apply_changes = False
            delete_shelf = False
        elif action == 'preview':
            apply_changes = False
            delete_shelf = False
            show_diff = True
        elif action == 'delete-only':
            apply_changes = False
            read_shelf = False
        elif action == 'keep':
            apply_changes = True
            delete_shelf = False
    except:
        tree.unlock()
        raise
    return klass(tree, manager, shelf_id, apply_changes, delete_shelf, read_shelf, show_diff, write_diff_to)