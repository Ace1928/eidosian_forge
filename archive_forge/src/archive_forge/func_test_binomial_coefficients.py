from sympy.ntheory.multinomial import (binomial_coefficients, binomial_coefficients_list, multinomial_coefficients)
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
def test_binomial_coefficients():
    for n in range(15):
        c = binomial_coefficients(n)
        l = [c[k] for k in sorted(c)]
        assert l == binomial_coefficients_list(n)