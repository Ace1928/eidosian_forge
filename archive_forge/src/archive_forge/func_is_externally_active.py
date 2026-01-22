from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
import sage.graphs.graph as graph
from sage.rings.rational_field import QQ
def is_externally_active(G, T, e):
    """
    Input:
    --A graph G.
    --A spanning tree T for G
    --And edge e of G

    Returns: ``True`` is e is not in T and e is externally active for T, ``False`` otherwise. Uses the ordering on G.edges()."""
    if T.has_edge(*e):
        return False
    for f in cyc(G, T, e):
        if edge_index(f) < edge_index(e):
            return False
    return True