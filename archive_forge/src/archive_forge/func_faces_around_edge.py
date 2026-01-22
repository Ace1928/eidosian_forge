import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def faces_around_edge(mcomplex, tet, edge):
    face = min(set(range(4)) - set(edge))
    ans = [(tet, face, edge)]
    A0 = arrow(mcomplex, tet, face, edge)
    A = A0.copy().next()
    while A != A0:
        tet = A.Tetrahedron.Index
        face = t3m.FaceIndex[A.Face]
        edge = t3m_edge_to_tuple[A.Edge]
        ans.append((tet, face, edge))
        A.next()
    return ans