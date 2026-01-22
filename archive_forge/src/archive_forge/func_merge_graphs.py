import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def merge_graphs(graph, *graphs, **kwargs):
    """Merges a bunch of graphs into a new graph.

    If no additional graphs are provided the first graph is
    returned unmodified otherwise the merged graph is returned.
    """
    tmp_graph = graph
    allow_overlaps = kwargs.get('allow_overlaps', False)
    overlap_detector = kwargs.get('overlap_detector')
    if overlap_detector is not None and (not callable(overlap_detector)):
        raise ValueError('Overlap detection callback expected to be callable')
    elif overlap_detector is None:
        overlap_detector = lambda to_graph, from_graph: len(to_graph.subgraph(from_graph.nodes))
    for g in graphs:
        if not allow_overlaps:
            overlaps = overlap_detector(graph, g)
            if overlaps:
                raise ValueError('Can not merge graph %s into %s since there are %s overlapping nodes (and we do not support merging nodes)' % (g, graph, overlaps))
        graph = nx.algorithms.compose(graph, g)
    if graphs:
        graph.name = tmp_graph.name
    return graph