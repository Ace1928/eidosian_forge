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
def remove_index(self, index):
    """Remove index from the indices used to answer queries.

        :param index: An index from the pack parameter.
        """
    del self.index_to_pack[index]
    pos = self.combined_index._indices.index(index)
    del self.combined_index._indices[pos]
    del self.combined_index._index_names[pos]
    if self.add_callback is not None and getattr(index, 'add_nodes', None) == self.add_callback:
        self.add_callback = None
        self.data_access.set_writer(None, None, (None, None))