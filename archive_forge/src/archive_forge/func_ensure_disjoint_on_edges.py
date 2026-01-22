from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def ensure_disjoint_on_edges(self):
    """
        Scales the cusp neighborhoods down until they are disjoint when
        intersected with the edges of the triangulations.

        Given an edge of a triangulation, we can easily compute the signed
        distance between the two cusp neighborhoods at the ends of the edge
        measured along that edge. Thus, we can easily check that all the
        distances measured along all the edges are positive and scale the
        cusps down if necessary.

        Unfortunately, this is not sufficient to ensure that two cusp
        neighborhoods are disjoint since there might be a geodesic between
        the two cusps such that the distance between the two cusps measured
        along the geodesic is shorter than measured along any edge of the
        triangulation.

        Thus, it is necessary to call ensure_std_form as well:
        it will make sure that the cusp neighborhoods are small enough so
        that they intersect the tetrahedra in "standard" form.
        Here, "standard" form means that the corresponding horoball about a
        vertex of a tetrahedron intersects the three faces of the tetrahedron
        adjacent to the vertex but not the one opposite to the vertex.

        For any geometric triangulation, standard form and positive distance
        measured along all edges of the triangulation is sufficient for
        disjoint neighborhoods.

        The SnapPea kernel uses the proto-canonical triangulation associated
        to the cusp neighborhood to get around this when computing the
        "reach" and the "stoppers" for the cusps.

        **Remark:** This means that the cusp neighborhoods might be scaled down
        more than necessary. Related open questions are: given maximal disjoint
        cusp neighborhoods (maximal in the sense that no neighborhood can be
        expanded without bumping into another or itself), is there always a
        geometric triangulation intersecting the cusp neighborhoods in standard
        form? Is there an easy algorithm to find this triangulation, e.g., by
        applying a 2-3 move whenever we see a non-standard intersection?
        """
    num_cusps = len(self.mcomplex.Vertices)
    for i in range(num_cusps):
        if (i, i) in self._edge_dict:
            dist = ComplexCuspCrossSection._exp_distance_of_edges(self._edge_dict[i, i])
            if not dist > 1:
                scale = sqrt(dist)
                ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[i], scale)
    for i in range(num_cusps):
        for j in range(i):
            if (j, i) in self._edge_dict:
                dist = ComplexCuspCrossSection._exp_distance_of_edges(self._edge_dict[j, i])
                if not dist > 1:
                    scale = sqrt(dist)
                    ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[i], scale)
                    ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[j], scale)