from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def enough_gluing_equations(manifold):
    """
    Select a full-rank portion of the gluing equations.
    """
    n_tet = manifold.num_tetrahedra()
    n_cusps = manifold.num_cusps()
    eqns = manifold.gluing_equations('rect')
    edge_eqns = pari_matrix([a + b for a, b, _ in eqns[:n_tet]])
    edge_eqns_with_RHS = pari_matrix([a + b + [(1 - c) // 2] for a, b, c in eqns[:n_tet]])
    H, U = edge_eqns.mattranspose().mathnf(flag=1)
    assert H.ncols() == n_tet - n_cusps
    edge_eqns_with_RHS = pari_matrix_to_lists(edge_eqns_with_RHS.mattranspose() * U)[n_cusps:]
    edge_eqns_with_RHS = [(e[:n_tet], e[n_tet:2 * n_tet], pari(-1) ** e[-1]) for e in edge_eqns_with_RHS]
    cusp_eqns = []
    j = n_tet
    for i in range(n_cusps):
        cusp_eqns.append(eqns[j])
        j += 2 if manifold.cusp_info(i)['complete?'] else 1
    ans_eqns = edge_eqns_with_RHS + cusp_eqns
    ans_matrix = pari_matrix([a + b for a, b, _ in ans_eqns])
    assert len(ans_eqns) == n_tet and len(ans_matrix.mattranspose().matkerint()) == 0
    return [(list(map(int, A)), list(map(int, B)), int(c)) for A, B, c in ans_eqns]