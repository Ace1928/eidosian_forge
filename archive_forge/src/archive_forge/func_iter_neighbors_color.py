import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def iter_neighbors_color(self, color):
    adj_color_node = self.adj_color[color]
    while adj_color_node is not None:
        yield adj_color_node.node_id
        adj_color_node = adj_color_node.col_next