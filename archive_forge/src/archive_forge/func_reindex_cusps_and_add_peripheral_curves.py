from .mcomplex_base import *
from .t3mlite import simplex
def reindex_cusps_and_add_peripheral_curves(self, cusp_indices_and_peripheral_curve_data):
    """
        Expects the result of
        Manifold._get_cusp_indices_and_peripheral_curve_data().

        It rearranges the Vertices of the mcomplex to match the ordering
        of the cusps in the SnapPea kernel and adds the peripheral curves
        in a format analogous to the kernel.
        """
    cusp_indices, curves = cusp_indices_and_peripheral_curve_data

    def process_row(curves):
        return {vertex: {face: curves[4 * i + j] for j, face in enumerate(simplex.TwoSubsimplices)} for i, vertex in enumerate(simplex.ZeroSubsimplices)}
    for i, tet in enumerate(self.mcomplex.Tetrahedra):
        tet.PeripheralCurves = [[process_row(curves[4 * i + 0]), process_row(curves[4 * i + 1])], [process_row(curves[4 * i + 2]), process_row(curves[4 * i + 3])]]
        for vertex, cusp_index in zip(simplex.ZeroSubsimplices, cusp_indices[i]):
            tet.Class[vertex].Index = cusp_index
    self.mcomplex.Vertices.sort(key=lambda vertex: vertex.Index)
    for index, vertex in enumerate(self.mcomplex.Vertices):
        if not index == vertex.Index:
            raise Exception('Inconsistencies with vertex indices')