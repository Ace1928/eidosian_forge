from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def insert_edge_loop(self, position):
    edge = self.loop[position]
    if edge.subcomplex_type != 'gamma':
        return position
    edge_index_and_end = self.truncated_complex.get_edge_index_and_end_from_tet_and_perm(edge.tet_and_perm)
    if edge_index_and_end not in self.uncovered_edge_ends:
        return position
    self.uncovered_edge_ends.remove(edge_index_and_end)
    self.loop.insert(position, TruncatedComplex.EdgeLoop(edge.tet_and_perm, edge_index_and_end[0]))
    return position + 1