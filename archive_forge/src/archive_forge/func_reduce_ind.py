import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def reduce_ind(ind, i):
    rind = ind[[k not in i for k in ind]]
    for k in set(i):
        rind[rind >= k] -= 1
    return rind