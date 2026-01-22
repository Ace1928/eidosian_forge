import math
from bisect import bisect_left
from itertools import accumulate, combinations, product
import networkx as nx
from networkx.utils import py_random_state
def should_join(pair):
    return seed.random() < beta * math.exp(-dist(*pair) / (alpha * L))