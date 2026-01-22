import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
def reload_pack_names(self):
    """Sync our pack listing with what is present in the repository.

        This should be called when we find out that something we thought was
        present is now missing. This happens when another process re-packs the
        repository, etc.

        :return: True if the in-memory list of packs has been altered at all.
        """
    first_read = self.ensure_loaded()
    if first_read:
        return True
    disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes = self._diff_pack_names()
    self._packs_at_load = orig_disk_nodes
    removed, added, modified = self._syncronize_pack_names_from_disk_nodes(disk_nodes)
    if removed or added or modified:
        return True
    return False