from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def cyc(G, T, e):
    """
    Input:
    --A graph G.
    --A spanning tree T for G
    --And edge e of G not in T

    Adjoining e to T creates a cycle.
    Returns: this cycle."""
    if not G.has_edge(*e):
        raise ValueError('e must be an edge of G.')
    if T.has_edge(*e):
        raise ValueError('e must not be an edge of T.')
    try:
        l = T.edge_label(e[0], e[1])
        if isinstance(l, list):
            l = l[0]
        if (e[0], e[1], l) in T.edges(sort=True, key=edge_index):
            return [(e[0], e[1], l), e]
        return [(e[1], e[0], l), e]
    except Exception:
        pass
    S = graph.Graph(T.edges(sort=True, key=edge_index))
    S.add_edge(e)
    cb = S.cycle_basis()[0]
    answer = list()
    for i in range(len(cb)):
        l = S.edge_label(cb[i], cb[(i + 1) % len(cb)])
        if S.has_edge(cb[i], cb[(i + 1) % len(cb)], l):
            answer.append((cb[i], cb[(i + 1) % len(cb)], l))
        else:
            answer.append((cb[(i + 1) % len(cb)], cb[i], l))
    return answer