from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def goeritz_matrix(self, return_graph=False):
    """
        Call self.white_graph() and return the Goeritz matrix of the result.
        If the return_graph flag is set, also return the graph::

            sage: K=Link('4_1')
            sage: abs(K.goeritz_matrix().det())
            5
        """
    G = self.white_graph()
    V = G.vertices(sort=True)
    N = len(V)
    m = matrix(N, N)
    vertex = {v: n for n, v in enumerate(V)}
    for e in G.edges(sort=False):
        i, j = (vertex[e[0]], vertex[e[1]])
        m[i, j] = m[j, i] = m[i, j] + e[2]['sign']
    for i in range(N):
        m[i, i] = -sum(m.column(i))
    m = m.delete_rows([0]).delete_columns([0])
    return (m, G) if return_graph else m