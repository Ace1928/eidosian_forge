import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def hash_regina_surface(S):
    T = S.getTriangulation()
    t = T.getNumberOfTetrahedra()
    ans = [S.getEulerCharacteristic()]
    ans += sorted([S.getEdgeWeight(i) for i in range(T.getNumberOfEdges())])
    ans += sorted([S.getQuadCoord(i, j) for i in range(t) for j in range(3)])
    return ans