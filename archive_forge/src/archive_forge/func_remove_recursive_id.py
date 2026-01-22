from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def remove_recursive_id(self, file_id):
    """Remove file_id, and children, from the inventory.

        :param file_id: A file_id to remove.
        """
    to_find_delete = [self._byid[file_id]]
    to_delete = []
    while to_find_delete:
        ie = to_find_delete.pop()
        to_delete.append(ie.file_id)
        if ie.kind == 'directory':
            to_find_delete.extend(ie.children.values())
    for file_id in reversed(to_delete):
        ie = self.get_entry(file_id)
        del self._byid[file_id]
    if ie.parent_id is not None:
        del self.get_entry(ie.parent_id).children[ie.name]
    else:
        self.root = None