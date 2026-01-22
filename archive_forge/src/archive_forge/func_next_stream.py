import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def next_stream():
    stream = source_vf.get_record_stream(cur_keys, 'as-requested', True)
    for record in stream:
        if record.storage_kind == 'absent':
            continue
        bytes = record.get_bytes_as('fulltext')
        node = chk_map._deserialise(bytes, record.key, search_key_func=None)
        common_base = node._search_prefix
        if isinstance(node, chk_map.InternalNode):
            handle_internal_node(node)
        elif parse_leaf_nodes:
            handle_leaf_node(node)
        counter[0] += 1
        if pb is not None:
            pb.update('chk node', counter[0], total_keys)
        yield record