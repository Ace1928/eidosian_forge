from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def edge_parameters_from_normalized_gram_matrices(mcomplex, gram_matrices):
    return [edge_parameter_from_normalized_gram_matrices(e, gram_matrices) for e in mcomplex.Edges]