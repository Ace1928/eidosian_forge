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
def get_pack_by_name(self, name):
    """Get a Pack object by name.

        :param name: The name of the pack - e.g. '123456'
        :return: A Pack object.
        """
    try:
        return self._packs_by_name[name]
    except KeyError:
        rev_index = self._make_index(name, '.rix')
        inv_index = self._make_index(name, '.iix')
        txt_index = self._make_index(name, '.tix')
        sig_index = self._make_index(name, '.six')
        if self.chk_index is not None:
            chk_index = self._make_index(name, '.cix', is_chk=True)
        else:
            chk_index = None
        result = ExistingPack(self._pack_transport, name, rev_index, inv_index, txt_index, sig_index, chk_index)
        self.add_pack_to_memory(result)
        return result