import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def sort_groupcompress(parent_map):
    """Sort and group the keys in parent_map into groupcompress order.

    groupcompress is defined (currently) as reverse-topological order, grouped
    by the key prefix.

    :return: A sorted-list of keys
    """
    from ..tsort import topo_sort
    per_prefix_map = {}
    for item in parent_map.items():
        key = item[0]
        if isinstance(key, bytes) or len(key) == 1:
            prefix = b''
        else:
            prefix = key[0]
        try:
            per_prefix_map[prefix].append(item)
        except KeyError:
            per_prefix_map[prefix] = [item]
    present_keys = []
    for prefix in sorted(per_prefix_map):
        present_keys.extend(reversed(topo_sort(per_prefix_map[prefix])))
    return present_keys