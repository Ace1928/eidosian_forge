import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
def check_is_well_utilized(self):
    """Is the current block considered 'well utilized'?

        This heuristic asks if the current block considers itself to be a fully
        developed group, rather than just a loose collection of data.
        """
    if len(self._factories) == 1:
        return False
    action, last_byte_used, total_bytes_used = self._check_rebuild_action()
    block_size = self._block._content_length
    if total_bytes_used < block_size * self._max_cut_fraction:
        return False
    if block_size >= self._full_enough_block_size:
        return True
    common_prefix = None
    for factory in self._factories:
        prefix = factory.key[:-1]
        if common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            if block_size >= self._full_enough_mixed_block_size:
                return True
            break
    return False