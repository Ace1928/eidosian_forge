from chempy.util.testing import requires
from ..integrated import pseudo_irrev, pseudo_rev, binary_irrev, binary_rev
@requires('sympy')
def test_binary_rev():
    f = binary_rev(t, kf, kb, prod, major, minor, backend=sympy)
    dfdt = f.diff(t)
    num_dfdt = dfdt.subs(subsd)
    ans = kf * (minor - f) * (major - f) - kb * f
    assert abs(float(num_dfdt) - float(ans.subs(subsd))) < 2e-14