from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RealDoubleField, RealIntervalField, vector, matrix, pi
def partially_verified_hyperbolic_structure(self, verbose=False):
    self.expand_until_certified(verbose)
    return HyperbolicStructure(self.mcomplex, self.certified_edge_lengths, self.exact_edges, self.var_edges)