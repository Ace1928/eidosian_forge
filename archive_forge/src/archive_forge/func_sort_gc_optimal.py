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
def sort_gc_optimal(parent_map):
    """Sort and group the keys in parent_map into groupcompress order.

    groupcompress is defined (currently) as reverse-topological order, grouped
    by the key prefix.

    :return: A sorted-list of keys
    """
    per_prefix_map = {}
    for key, value in parent_map.items():
        if isinstance(key, bytes) or len(key) == 1:
            prefix = b''
        else:
            prefix = key[0]
        try:
            per_prefix_map[prefix][key] = value
        except KeyError:
            per_prefix_map[prefix] = {key: value}
    present_keys = []
    for prefix in sorted(per_prefix_map):
        present_keys.extend(reversed(tsort.topo_sort(per_prefix_map[prefix])))
    return present_keys