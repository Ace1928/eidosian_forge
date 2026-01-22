from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
def get_incoming_edges(downstream_node, incoming_edge_map):
    edges = []
    for downstream_label, upstream_info in list(incoming_edge_map.items()):
        upstream_node, upstream_label, upstream_selector = upstream_info
        edges += [DagEdge(downstream_node, downstream_label, upstream_node, upstream_label, upstream_selector)]
    return edges