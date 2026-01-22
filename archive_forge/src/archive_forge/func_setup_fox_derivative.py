import string
from ..sage_helper import _within_sage, sage_method
def setup_fox_derivative(word, phi, var, involute=False):
    R = phi.range()
    if len(word) == 0:
        return R.zero()
    gens = list(set((var + word).lower()))
    gens += [g.upper() for g in gens]
    phi_ims = {}
    fox_ders = {}
    for g in gens:
        phi_ims[g] = phi(g) if not involute else phi(g.swapcase())
        if g == g.lower():
            fox_ders[g] = R.zero() if g != var else R.one()
        else:
            fox_ders[g] = R.zero() if g.lower() != var else -phi_ims[var.upper()]
    return (R, phi_ims, fox_ders)