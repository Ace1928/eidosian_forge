from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
@staticmethod
def describe_change(old_entry, new_entry):
    """Describe the change between old_entry and this.

        This smells of being an InterInventoryEntry situation, but as its
        the first one, we're making it a static method for now.

        An entry with a different parent, or different name is considered
        to be renamed. Reparenting is an internal detail.
        Note that renaming the parent does not trigger a rename for the
        child entry itself.
        """
    if old_entry is new_entry:
        return 'unchanged'
    elif old_entry is None:
        return 'added'
    elif new_entry is None:
        return 'removed'
    if old_entry.kind != new_entry.kind:
        return 'modified'
    text_modified, meta_modified = new_entry.detect_changes(old_entry)
    if text_modified or meta_modified:
        modified = True
    else:
        modified = False
    if old_entry.parent_id != new_entry.parent_id:
        renamed = True
    elif old_entry.name != new_entry.name:
        renamed = True
    else:
        renamed = False
    if renamed and (not modified):
        return InventoryEntry.RENAMED
    if modified and (not renamed):
        return 'modified'
    if modified and renamed:
        return InventoryEntry.MODIFIED_AND_RENAMED
    return 'unchanged'