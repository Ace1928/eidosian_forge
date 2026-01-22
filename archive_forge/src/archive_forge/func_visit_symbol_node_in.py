from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
def visit_symbol_node_in(self, node):
    graph_node_id = str(id(node))
    graph_node_label = repr(node)
    graph_node_color = 8421504
    graph_node_style = '"filled"'
    if node.is_intermediate:
        graph_node_shape = 'ellipse'
    else:
        graph_node_shape = 'rectangle'
    graph_node = self.pydot.Node(graph_node_id, style=graph_node_style, fillcolor='#{:06x}'.format(graph_node_color), shape=graph_node_shape, label=graph_node_label)
    self.graph.add_node(graph_node)
    return iter(node.children)