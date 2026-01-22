from .truncatedComplex import TruncatedComplex
from snappy.snap import t3mlite as t3m
from .verificationError import *
def mark_hexagon_as_used(self, edges):
    for edge in edges:
        if edge.subcomplex_type == 'beta':
            tet_index, p = edge.tet_and_perm
            self.used_hexagons.add((tet_index, p.tuple()))