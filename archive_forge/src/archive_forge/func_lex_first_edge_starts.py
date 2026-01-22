import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def lex_first_edge_starts(mcomplex):
    """
    Returns a list of containing tuples of the form (tet, face, edge),
    one at each edge.
    """
    T = mcomplex
    ans = []
    for edge in T.Edges:
        poss_starts = []
        for C in edge.Corners:
            t = C.Tetrahedron.Index
            e = t3m_edge_to_tuple[C.Subsimplex]
            poss_starts.append((t, e))
        ans.append(min(poss_starts))
    return sorted(ans)