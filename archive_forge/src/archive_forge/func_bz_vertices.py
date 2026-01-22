import warnings
from math import pi, sin, cos
import numpy as np
def bz_vertices(icell, dim=3):
    """See https://xkcd.com/1421 ..."""
    from scipy.spatial import Voronoi
    icell = icell.copy()
    if dim < 3:
        icell[2, 2] = 0.001
    if dim < 2:
        icell[1, 1] = 0.001
    I = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
    G = np.dot(icell.T, I).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 13 in points:
            normal = G[points].sum(0)
            normal /= (normal ** 2).sum() ** 0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1