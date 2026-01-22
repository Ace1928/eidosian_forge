from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_chain_criterion():
    gens = [x]
    f1 = sdm_from_vector([1, x], grlex, QQ, gens=gens)
    f2 = sdm_from_vector([0, x - 2], grlex, QQ, gens=gens)
    assert len(sdm_groebner([f1, f2], sdm_nf_mora, grlex, QQ)) == 2