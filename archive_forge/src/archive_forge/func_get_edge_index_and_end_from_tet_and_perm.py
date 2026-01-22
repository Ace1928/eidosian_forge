from snappy.snap import t3mlite as t3m
def get_edge_index_and_end_from_tet_and_perm(self, tet_and_perm):
    tet_index, p = tet_and_perm
    tet = self.mcomplex.Tetrahedra[tet_index]
    return (tet.Class[p.image(t3m.E01)].Index, self.tet_and_perm_to_end_of_edge[tet_index, p.tuple()])