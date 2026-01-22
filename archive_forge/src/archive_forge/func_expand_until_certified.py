from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RealDoubleField, RealIntervalField, vector, matrix, pi
def expand_until_certified(self, verbose=False):
    try:
        h = HyperbolicStructure(self.mcomplex, self.initial_edge_lengths, self.exact_edges, self.var_edges)
    except BadDihedralAngleError as e:
        raise KrawczykFailedWithBadDihedralAngleError('When preparing for certification', e)
    error_at_initial_edge_lengths = vector(self.RIF, [h.angle_sums[e] - self.twoPi for e in self.var_edges])
    self.first_term = vector(self.RIF, [self.initial_edge_lengths[c] for c in self.var_edges]) - self.approx_inverse * error_at_initial_edge_lengths
    edge_lengths = self.initial_edge_lengths
    num_iterations = 25 if self.bits_prec > 53 else 11
    for i in range(num_iterations + 1):
        old_edge_lengths = edge_lengths
        edge_lengths = self.krawczyk_iteration(edge_lengths)
        if KrawczykCertifiedEdgeLengthsEngine.interval_vector_is_contained_in(edge_lengths, old_edge_lengths):
            self.certified_edge_lengths = edge_lengths
            if verbose:
                print('Certified in iteration', i)
            return True
        edge_lengths = KrawczykCertifiedEdgeLengthsEngine.interval_vector_union(edge_lengths, old_edge_lengths)
    raise KrawczykFailedToFinishError('Failed after iterations', num_iterations)