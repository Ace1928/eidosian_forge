from .computeApproxHyperbolicStructureNew import *
from .computeApproxHyperbolicStructureOrb import *
from .polishApproxHyperbolicStructure import *
from .krawczykCertifiedEdgeLengthsEngine import *
from .verifyHyperbolicStructureEngine import *
from .parseVertexGramMatrixFile import (
from snappy.snap.t3mlite import Mcomplex
def compute_unverified_hyperbolic_structure(triangulation, source='new', verbose=False):
    """
    Given a snappy.Triangulation, computes an unverified hyperbolic structure,
    i.e., an instance of HyperbolicStructure where the edge lengths are in
    SageMath's RealDoubleField.

    The optional argument source can be:
       - 'new' to use the new python only implementation
       - 'orb' to use Orb
       - the path to a file containing vertex gram matrices as produced by
         orb_solution_for_snappea_finite_triangulation
    """
    if source == 'new':
        return compute_approx_hyperbolic_structure_new(Mcomplex(triangulation), verbose=verbose)
    elif source == 'orb':
        return compute_approx_hyperbolic_structure_orb(triangulation)
    else:
        return compute_approx_hyperbolic_structure_from_vertex_gram_matrix_file(Mcomplex(triangulation), source)