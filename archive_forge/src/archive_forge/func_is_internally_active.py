from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def is_internally_active(G, T, e):
    """
    Input:
    --A graph G.
    --A spanning tree T for G
    --And edge e of G

    Returns: ``True`` if e is in T and e is internally active for T, ``False`` otherwise. Uses the ordering on G.edges()."""
    if not T.has_edge(*e):
        return False
    for f in cut(G, T, e):
        if edge_index(f) < edge_index(e):
            return False
    return True