from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def edge_parameters_from_gram_matrices(mcomplex, gram_matrices):
    return vector(RealDoubleField(), edge_parameters_from_normalized_gram_matrices(mcomplex, normalize_gram_matrices(gram_matrices)))