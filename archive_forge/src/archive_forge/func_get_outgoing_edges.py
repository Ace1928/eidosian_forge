from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
def get_outgoing_edges(upstream_node, outgoing_edge_map):
    edges = []
    for upstream_label, downstream_infos in sorted(outgoing_edge_map.items()):
        for downstream_info in downstream_infos:
            downstream_node, downstream_label, downstream_selector = downstream_info
            edges += [DagEdge(downstream_node, downstream_label, upstream_node, upstream_label, downstream_selector)]
    return edges