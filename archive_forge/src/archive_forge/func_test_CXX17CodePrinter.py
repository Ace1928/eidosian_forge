from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p
def test_CXX17CodePrinter():
    assert CXX17CodePrinter().doprint(beta(x, y)) == 'std::beta(x, y)'
    assert CXX17CodePrinter().doprint(Ei(x)) == 'std::expint(x)'
    assert CXX17CodePrinter().doprint(zeta(x)) == 'std::riemann_zeta(x)'
    assert CXX17CodePrinter().doprint(frac(x)) == 'x - std::floor(x)'
    assert CXX17CodePrinter().doprint(riemann_xi(x)) == '(1.0/2.0)*std::pow(M_PI, -1.0/2.0*x)*x*(x - 1)*std::tgamma((1.0/2.0)*x)*std::riemann_zeta(x)'