from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RealDoubleField, RealIntervalField, vector, matrix, pi
def krawczyk_iteration(self, edge_lengths):
    try:
        h = HyperbolicStructure(self.mcomplex, edge_lengths, self.exact_edges, self.var_edges)
    except BadDihedralAngleError as e:
        raise KrawczykFailedWithBadDihedralAngleError('During iteration', e)
    error = [h.angle_sums[e] - self.twoPi for e in self.var_edges]
    jacobian = h.full_rank_jacobian_submatrix()
    diffs = vector(self.RIF, [edge_lengths[var_edge] - self.initial_edge_lengths[var_edge] for var_edge in self.var_edges])
    var_edge_lengths = self.first_term + (self.identity - self.approx_inverse * jacobian) * diffs
    result = [edge_length for edge_length in edge_lengths]
    for var_edge, edge_length in zip(self.var_edges, var_edge_lengths):
        result[var_edge] = edge_length
    return vector(self.RIF, result)