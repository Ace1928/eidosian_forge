from snappy.snap import t3mlite as t3m
@staticmethod
def get_tet_and_odd_perms_for_vertex(vertex):
    for corner in vertex.Corners:
        for perm in t3m.Perm4.A4():
            p = perm * t3m.Perm4([0, 1, 3, 2])
            if p.image(t3m.V0) == corner.Subsimplex:
                yield (corner.Tetrahedron.Index, p)