from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def lcm_divides(ip):
    m = monomial_lcm(mh, f[ip].LM)
    return monomial_div(LCMhg, m)