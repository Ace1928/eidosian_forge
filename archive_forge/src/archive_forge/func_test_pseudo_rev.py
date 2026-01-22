from chempy.util.testing import requires
from ..integrated import pseudo_irrev, pseudo_rev, binary_irrev, binary_rev
@requires('sympy')
def test_pseudo_rev():
    f = pseudo_rev(t, kf, kb, prod, major, minor, backend=sympy)
    dfdt = f.diff(t)
    num_dfdt = dfdt.subs(subsd)
    assert (num_dfdt - (major * kf * (minor - f) - kb * f).subs(subsd)).simplify() == 0