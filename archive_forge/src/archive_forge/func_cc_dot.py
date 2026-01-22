import itertools
import networkx as nx
def cc_dot(nu, nv):
    return len(nu & nv) / len(nu | nv)