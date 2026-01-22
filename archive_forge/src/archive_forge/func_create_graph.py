import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def create_graph(objects, *, context=None, filter=None):
    if context is None:
        context = cuda_allocation_context()
    if filter is None:
        filter = is_cuda_tensor
    nodes = [Node(object_annotation(obj), context(obj), filter(obj), []) for obj in objects]
    node_referrers: List[List[int]] = [[] for obj in objects]
    id_to_node = {id(obj): i for i, obj in enumerate(objects)}
    for obj in objects:
        fidx = id_to_node[id(obj)]
        f = nodes[fidx]
        references = annotated_references(obj)
        for referrent in gc.get_referents(obj):
            rid = id(referrent)
            tidx = id_to_node.get(rid, None)
            if tidx is None:
                continue
            t = nodes[tidx]
            labels = references.get(rid, ['?'])
            node_referrers[tidx].append(fidx)
            for label in labels:
                f.referrents.append((label, tidx))
    to_search = [i for i, n in enumerate(nodes) if n.root]
    to_keep = set()
    while to_search:
        idx = to_search.pop()
        if idx in to_keep:
            continue
        to_keep.add(idx)
        referrers = node_referrers[idx]
        to_search.extend(referrers)
    id_to_filtered_id: Dict[int, int] = {}
    filtered: List[Any] = []
    for i, n in enumerate(nodes):
        if i in to_keep:
            id_to_filtered_id[i] = len(id_to_filtered_id)
            filtered.append(n)
    for n in filtered:
        n.referrents[:] = [(label, id_to_filtered_id[idx]) for label, idx in n.referrents if idx in id_to_filtered_id]
    return filtered