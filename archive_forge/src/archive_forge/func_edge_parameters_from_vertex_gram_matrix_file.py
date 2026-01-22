from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def edge_parameters_from_vertex_gram_matrix_file(mcomplex, filename):
    return edge_parameters_from_gram_matrices(mcomplex, eval(open(filename).read()))