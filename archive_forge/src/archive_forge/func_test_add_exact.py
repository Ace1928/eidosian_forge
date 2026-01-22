from mpmath import *
from mpmath.libmp import *
def test_add_exact():
    ff = from_float
    assert mpf_add(ff(3.0), ff(2.5)) == ff(5.5)
    assert mpf_add(ff(3.0), ff(-2.5)) == ff(0.5)
    assert mpf_add(ff(-3.0), ff(2.5)) == ff(-0.5)
    assert mpf_add(ff(-3.0), ff(-2.5)) == ff(-5.5)
    assert mpf_sub(mpf_add(fone, ff(1e-100)), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(ff(1e-100), fone), fone) == ff(1e-100)
    assert mpf_sub(mpf_add(fone, ff(-1e-100)), fone) == ff(-1e-100)
    assert mpf_sub(mpf_add(ff(-1e-100), fone), fone) == ff(-1e-100)
    assert mpf_add(fone, fzero) == fone
    assert mpf_add(fzero, fone) == fone
    assert mpf_add(fzero, fzero) == fzero