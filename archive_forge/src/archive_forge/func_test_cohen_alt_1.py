from mpmath import mp
from mpmath import libmp
def test_cohen_alt_1():
    mp.dps = 17
    A = []
    AC = mp.cohen_alt()
    n = 1
    while 1:
        A.append(mp.loggamma(1 + mp.one / (2 * n - 1)))
        A.append(-mp.loggamma(1 + mp.one / (2 * n)))
        n += 1
        v, e = AC.update(A)
        if e < mp.eps:
            break
        if n > 1000:
            raise RuntimeError('iteration limit exceeded')
    v = mp.exp(v)
    err = abs(v - 1.06215090557106)
    assert err < 1e-12