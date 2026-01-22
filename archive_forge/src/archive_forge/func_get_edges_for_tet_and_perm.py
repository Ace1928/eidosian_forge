from snappy.snap import t3mlite as t3m
def get_edges_for_tet_and_perm(self, tet_and_perm):
    """
        Given a vertex in the truncated complex parametrized by
        (index of tet, perm), return all four edges starting at that
        vertex.
        """
    other_tet_and_perm = self.get_glued_tet_and_perm(tet_and_perm)
    return [TruncatedComplex.Edge('alpha', tet_and_perm), TruncatedComplex.Edge('beta', tet_and_perm), TruncatedComplex.Edge('gamma', tet_and_perm), TruncatedComplex.Edge('gamma', other_tet_and_perm)]