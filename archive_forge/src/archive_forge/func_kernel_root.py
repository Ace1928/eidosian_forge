import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
def kernel_root(y, a, r):

    def my_function(b):
        return kernel_integral(y, a, b) - r
    return sp.optimize.brentq(my_function, a, 1)