from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
def randomize_graph(size, g):
    p = list(range(size))
    random.shuffle(p)
    g1a = {}
    for k, v in g1.items():
        g1a[p[k]] = [p[i] for i in v]
    return g1a