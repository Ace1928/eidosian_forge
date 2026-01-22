from chempy.util.testing import requires
from ..integrated import pseudo_irrev, pseudo_rev, binary_irrev, binary_rev
@requires('sympy')
def test_binary_irrev():
    f = binary_irrev(t, kf, prod, major, minor, backend=sympy)
    dfdt = f.diff(t)
    num_dfdt = dfdt.subs(subsd)
    assert (num_dfdt - (kf * (minor - f) * (major - f)).subs(subsd)).simplify() == 0