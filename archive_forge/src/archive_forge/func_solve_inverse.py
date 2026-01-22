import networkx as nx
def solve_inverse(self, r):
    rhs = np.zeros(self.n, self.dtype)
    rhs[r] = 1
    return sp.sparse.linalg.cg(self.L1, rhs[1:], M=self.M, atol=0)[0]