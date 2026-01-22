from collections import deque
from ..lazy_import import lazy_import
from breezy.bzr import (
from .. import errors, lazy_regex, osutils, trace
from .static_tuple import StaticTuple
def parent_candidates(self, previous_inventories):
    """Find possible per-file graph parents.

        This is currently defined by:
         - Select the last changed revision in the parent inventory.
         - Do deal with a short lived bug in bzr 0.8's development two entries
           that have the same last changed but different 'x' bit settings are
           changed in-place.
        """
    candidates = {}
    for inv in previous_inventories:
        try:
            ie = inv.get_entry(self.file_id)
        except errors.NoSuchId:
            pass
        else:
            if ie.revision in candidates:
                try:
                    if candidates[ie.revision].executable != ie.executable:
                        candidates[ie.revision].executable = False
                        ie.executable = False
                except AttributeError:
                    pass
            else:
                candidates[ie.revision] = ie
    return candidates