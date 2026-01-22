import itertools
import networkx as nx
def cc_min(nu, nv):
    return len(nu & nv) / min(len(nu), len(nv))