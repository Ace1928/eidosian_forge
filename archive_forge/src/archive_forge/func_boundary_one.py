from .simplex import *
from .linalg import Matrix
def boundary_one(manifold):
    V, E = (len(manifold.Vertices), len(manifold.Edges))
    ans = Matrix(V, E)
    for e in manifold.Edges:
        v_init, v_term = [v.Index for v in e.Vertices]
        ans[v_term, e.Index] += 1
        ans[v_init, e.Index] += -1
    return ans