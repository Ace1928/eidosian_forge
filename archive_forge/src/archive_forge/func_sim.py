import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def sim(u, v):
    return importance_factor * avg_sim(list(product(Gadj[u], Gadj[v])))