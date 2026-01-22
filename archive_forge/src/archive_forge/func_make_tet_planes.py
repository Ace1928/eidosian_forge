from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def make_tet_planes(tet_vert_positions):
    """
    Given four light-like vectors, returns the four normals for the
    for faces of the ideal tetrahedron spanned by the corresponding
    ideal points in the 1,3-hyperboloid model.

    Outward facing for positively oriented tetrahedra.
    """
    v0, v1, v2, v3 = tet_vert_positions
    return [R13_plane_from_R13_light_vectors([v1, v3, v2]), R13_plane_from_R13_light_vectors([v0, v2, v3]), R13_plane_from_R13_light_vectors([v0, v3, v1]), R13_plane_from_R13_light_vectors([v0, v1, v2])]