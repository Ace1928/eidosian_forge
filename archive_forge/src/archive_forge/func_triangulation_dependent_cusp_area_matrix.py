from ...sage_helper import _within_sage, sage_method
from ..cuspCrossSection import ComplexCuspCrossSection
from ..shapes import compute_hyperbolic_shapes
from ...math_basics import correct_min
from .cusp_tiling_engine import *
def triangulation_dependent_cusp_area_matrix(snappy_manifold, verified, bits_prec=None):
    """
    Interesting case: t12521

    Maximal cusp area matrix:

    [ 77.5537626509970512653317518641810890989543820290380458409? 11.40953140648583915022197187043644048603871960228564151087?]
    [11.40953140648583915022197187043644048603871960228564151087?     91.1461442179608339668518063027198489593908228325190920?]

    This result:

    [  77.553762651?   11.409531407?]
    [  11.409531407? 5.508968850234?]

    After M.canonize:

    [  62.42018359?  11.409531407?]
    [ 11.409531407? 15.1140644993?]
    """
    shapes = compute_hyperbolic_shapes(snappy_manifold, verified=verified, bits_prec=bits_prec)
    c = ComplexCuspCrossSection.fromManifoldAndShapes(snappy_manifold, shapes)
    c.ensure_std_form(allow_scaling_up=True)
    areas = c.cusp_areas()
    RIF = areas[0].parent()

    def entry(i, j):
        if i > j:
            i, j = (j, i)
        result = areas[i] * areas[j]
        if (i, j) in c._edge_dict:
            result *= correct_min([RIF(1), ComplexCuspCrossSection._exp_distance_of_edges(c._edge_dict[i, j])]) ** 2
        return result
    return _to_matrix([[entry(i, j) for i in range(len(areas))] for j in range(len(areas))])