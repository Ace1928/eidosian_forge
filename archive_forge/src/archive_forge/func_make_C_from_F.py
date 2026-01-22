from collections import defaultdict
import networkx as nx
def make_C_from_F(F):
    C = defaultdict(list)
    for node, color in F.items():
        C[color].append(node)
    return C