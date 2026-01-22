from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def barycentric_quad_embedding0(arrow, north_pole=None):
    """
    Take an arrow with 4 valent axis, then build embedding of the
    surrounding four tetrahedra forming an octahedron in R^3 using the
    arrows running around the valence 4 edge in xy plane.
    """
    n = Vector3([0, 0, 1]) if north_pole is None else north_pole
    e = Vector3([1, 0, 0])
    s = Vector3([0, 0, -1])
    w = Vector3([-1, 0, 0])
    a = Vector3([0, -1, 0])
    b = Vector3([0, 1, 0])
    arrow = arrow.copy()
    ans = []
    verts = [e, b, w, a, e]
    for i in range(4):
        bdry_map = [None, None, f'x{i}', f'y{i}']
        tet_verts = [verts[i], verts[i + 1], s, n]
        ans.append((arrow.Tetrahedron, tetrahedron_embedding(arrow, tet_verts, bdry_map)))
        arrow.next()
    return ans