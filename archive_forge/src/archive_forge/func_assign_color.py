import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
def assign_color(self, adj_entry, color):
    adj_entry.col_prev = None
    adj_entry.col_next = self.adj_color[color]
    self.adj_color[color] = adj_entry
    if adj_entry.col_next is not None:
        adj_entry.col_next.col_prev = adj_entry