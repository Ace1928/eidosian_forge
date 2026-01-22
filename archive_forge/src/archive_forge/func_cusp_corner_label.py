import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
def cusp_corner_label(v, w):
    return TruncatedSimplexCorners[v].index(v | w)