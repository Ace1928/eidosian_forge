from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def twisted_alexander_polynomial(alpha, reduced=False):
    """
    In HKL, alpha is epsilon x rho; in nsagetools, it would be called
    phialpha with phi being epsilon.  If reduced is True, the answer is
    divided by (t - 1).

    Here, we duplicate the calculation of Section 10.2 of [HKL].

       sage: M = Manifold('K12a169')
       sage: G = M.fundamental_group()
       sage: A = matrix(GF(5), [[0, 4], [1, 4]])
       sage: rho = cyclic_rep(G, A)
       sage: chi = lambda v:3*v[0]
       sage: alpha = induced_rep_from_twisted_cocycle(3, rho, chi, (0, 0, 1, 0))
       sage: -twisted_alexander_polynomial(alpha, reduced=True)
       4*t^2 + (z^3 + z^2 + 5)*t + 4
    """
    F = alpha('a').base_ring().base_ring()
    epsilon = alpha.epsilon
    gens, rels = (alpha.generators, alpha.relators)
    k = len(gens)
    assert len(rels) == len(gens) - 1 and epsilon.range().rank() == 1
    i0 = next((i for i, g in enumerate(gens) if epsilon(g) != 0))
    gens = gens[i0:] + gens[:i0]
    d2 = [[fox_derivative_with_involution(R, alpha, g) for R in rels] for g in gens]
    d2 = block_matrix(d2, nrows=k, ncols=k - 1)
    d1 = [alpha(g.swapcase()) - 1 for g in gens]
    d1 = block_matrix(d1, nrows=1, ncols=k)
    assert d1 * d2 == 0
    T = last_square_submatrix(d2)
    B = first_square_submatrix(d1)
    T = normalize_polynomial(fast_determinant_of_laurent_poly_matrix(T))
    B = normalize_polynomial(fast_determinant_of_laurent_poly_matrix(B))
    q, r = T.quo_rem(B)
    assert r == 0
    ans = normalize_polynomial(q)
    if reduced:
        t = ans.parent().gen()
        ans, r = ans.quo_rem(t - 1)
        assert r == 0
    return ans