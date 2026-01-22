from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def iter_all_entries(self):
    """Iterate over all keys within the index.

        :return: An iterable of (index, key, value) or
            (index, key, value, reference_lists).
            The former tuple is used when there are no reference lists in the
            index, making the API compatible with simple key:value index types.
            There is no defined order for the result iteration - it will be in
            the most efficient order for the index.
        """
    if 'evil' in debug.debug_flags:
        trace.mutter_callsite(3, 'iter_all_entries scales with size of history.')
    if not self.key_count():
        return
    if self._row_offsets[-1] == 1:
        if self.node_ref_lists:
            for key, (value, refs) in self._root_node.all_items():
                yield (self, key, value, refs)
        else:
            for key, (value, refs) in self._root_node.all_items():
                yield (self, key, value)
        return
    start_of_leaves = self._row_offsets[-2]
    end_of_leaves = self._row_offsets[-1]
    needed_offsets = list(range(start_of_leaves, end_of_leaves))
    if needed_offsets == [0]:
        nodes = [(0, self._root_node)]
    else:
        nodes = self._read_nodes(needed_offsets)
    if self.node_ref_lists:
        for _, node in nodes:
            for key, (value, refs) in node.all_items():
                yield (self, key, value, refs)
    else:
        for _, node in nodes:
            for key, (value, refs) in node.all_items():
                yield (self, key, value)