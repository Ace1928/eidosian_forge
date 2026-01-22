import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def to_digraph(self):
    """Converts this node + its children into a ordered directed graph.

        The graph returned will have the same structure as the
        this node and its children (and tree node metadata will be translated
        into graph node metadata).

        :returns: a directed graph
        :rtype: :py:class:`taskflow.types.graph.OrderedDiGraph`
        """
    g = graph.OrderedDiGraph()
    for node in self.bfs_iter(include_self=True, right_to_left=True):
        g.add_node(node.item, **node.metadata)
        if node is not self:
            g.add_edge(node.parent.item, node.item)
    return g