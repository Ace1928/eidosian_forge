from snappy.snap import t3mlite as t3m
@staticmethod
def get_edges_of_rectangle(tet_and_perm):
    tet_index, p = tet_and_perm
    return [TruncatedComplex.Edge('gamma', tet_and_perm), TruncatedComplex.Edge('alpha', (tet_index, p * t3m.Perm4([0, 1, 3, 2]))), TruncatedComplex.Edge('gamma', (tet_index, p * t3m.Perm4([1, 0, 3, 2]))), TruncatedComplex.Edge('alpha', (tet_index, p * t3m.Perm4([1, 0, 2, 3])))]