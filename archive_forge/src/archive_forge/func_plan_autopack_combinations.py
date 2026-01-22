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
def plan_autopack_combinations(self, existing_packs, pack_distribution):
    """Plan a pack operation.

        :param existing_packs: The packs to pack. (A list of (revcount, Pack)
            tuples).
        :param pack_distribution: A list with the number of revisions desired
            in each pack.
        """
    if len(existing_packs) <= len(pack_distribution):
        return []
    existing_packs.sort(reverse=True)
    pack_operations = [[0, []]]
    while len(existing_packs):
        next_pack_rev_count, next_pack = existing_packs.pop(0)
        if next_pack_rev_count >= pack_distribution[0]:
            while next_pack_rev_count > 0:
                next_pack_rev_count -= pack_distribution[0]
                if next_pack_rev_count >= 0:
                    del pack_distribution[0]
                else:
                    pack_distribution[0] = -next_pack_rev_count
        else:
            pack_operations[-1][0] += next_pack_rev_count
            pack_operations[-1][1].append(next_pack)
            if pack_operations[-1][0] >= pack_distribution[0]:
                del pack_distribution[0]
                pack_operations.append([0, []])
    final_rev_count = 0
    final_pack_list = []
    for num_revs, pack_files in pack_operations:
        final_rev_count += num_revs
        final_pack_list.extend(pack_files)
    if len(final_pack_list) == 1:
        raise AssertionError('We somehow generated an autopack with a single pack file being moved.')
        return []
    return [[final_rev_count, final_pack_list]]