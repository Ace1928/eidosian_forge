from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def build_bounding_info(self, manifold):
    if self.type() != 'normal':
        return (0, 0, None)
    bounds_subcomplex = 1
    double_bounds_subcomplex = 1
    for w in self.EdgeWeights:
        if w != 0 and w != 2:
            bounds_subcomplex = 0
        if w != 0 and w != 1:
            double_bounds_subcomplex = 0
        if not (bounds_subcomplex or double_bounds_subcomplex):
            break
    if bounds_subcomplex or double_bounds_subcomplex:
        thick_or_thin = 'thin'
        for tet in manifold.Tetrahedra:
            inside = 1
            for e in OneSubsimplices:
                w = self.get_edge_weight(tet.Class[e])
                if w != 0:
                    inside = 0
                    break
            if inside:
                thick_or_thin = 'thick'
                break
    else:
        thick_or_thin = None
    self.BoundingInfo = (bounds_subcomplex, double_bounds_subcomplex, thick_or_thin)