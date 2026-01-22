from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def compute_approx_hyperbolic_structure_from_vertex_gram_matrix_file(mcomplex, filename):
    return HyperbolicStructure(mcomplex, edge_parameters_from_vertex_gram_matrix_file(mcomplex, filename))