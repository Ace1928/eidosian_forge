from mpmath import *
from mpmath.libmp import *
import random
def test_pow_epsilon_rounding():
    """
    Stress test directed rounding for powers with integer exponents.
    Basically, we look at the following cases:

    >>> 1.0001 ** -5 # doctest: +SKIP
    0.99950014996500702
    >>> 0.9999 ** -5 # doctest: +SKIP
    1.000500150035007
    >>> (-1.0001) ** -5 # doctest: +SKIP
    -0.99950014996500702
    >>> (-0.9999) ** -5 # doctest: +SKIP
    -1.000500150035007

    >>> 1.0001 ** -6 # doctest: +SKIP
    0.99940020994401269
    >>> 0.9999 ** -6 # doctest: +SKIP
    1.0006002100560125
    >>> (-1.0001) ** -6 # doctest: +SKIP
    0.99940020994401269
    >>> (-0.9999) ** -6 # doctest: +SKIP
    1.0006002100560125

    etc.

    We run the tests with values a very small epsilon away from 1:
    small enough that the result is indistinguishable from 1 when
    rounded to nearest at the output precision. We check that the
    result is not erroneously rounded to 1 in cases where the
    rounding should be done strictly away from 1.
    """

    def powr(x, n, r):
        return make_mpf(mpf_pow_int(x._mpf_, n, mp.prec, r))
    for inprec, outprec in [(100, 20), (5000, 3000)]:
        mp.prec = inprec
        pos10001 = mpf(1) + mpf(2) ** (-inprec + 5)
        pos09999 = mpf(1) - mpf(2) ** (-inprec + 5)
        neg10001 = -pos10001
        neg09999 = -pos09999
        mp.prec = outprec
        r = round_up
        assert powr(pos10001, 5, r) > 1
        assert powr(pos09999, 5, r) == 1
        assert powr(neg10001, 5, r) < -1
        assert powr(neg09999, 5, r) == -1
        assert powr(pos10001, 6, r) > 1
        assert powr(pos09999, 6, r) == 1
        assert powr(neg10001, 6, r) > 1
        assert powr(neg09999, 6, r) == 1
        assert powr(pos10001, -5, r) == 1
        assert powr(pos09999, -5, r) > 1
        assert powr(neg10001, -5, r) == -1
        assert powr(neg09999, -5, r) < -1
        assert powr(pos10001, -6, r) == 1
        assert powr(pos09999, -6, r) > 1
        assert powr(neg10001, -6, r) == 1
        assert powr(neg09999, -6, r) > 1
        r = round_down
        assert powr(pos10001, 5, r) == 1
        assert powr(pos09999, 5, r) < 1
        assert powr(neg10001, 5, r) == -1
        assert powr(neg09999, 5, r) > -1
        assert powr(pos10001, 6, r) == 1
        assert powr(pos09999, 6, r) < 1
        assert powr(neg10001, 6, r) == 1
        assert powr(neg09999, 6, r) < 1
        assert powr(pos10001, -5, r) < 1
        assert powr(pos09999, -5, r) == 1
        assert powr(neg10001, -5, r) > -1
        assert powr(neg09999, -5, r) == -1
        assert powr(pos10001, -6, r) < 1
        assert powr(pos09999, -6, r) == 1
        assert powr(neg10001, -6, r) < 1
        assert powr(neg09999, -6, r) == 1
        r = round_ceiling
        assert powr(pos10001, 5, r) > 1
        assert powr(pos09999, 5, r) == 1
        assert powr(neg10001, 5, r) == -1
        assert powr(neg09999, 5, r) > -1
        assert powr(pos10001, 6, r) > 1
        assert powr(pos09999, 6, r) == 1
        assert powr(neg10001, 6, r) > 1
        assert powr(neg09999, 6, r) == 1
        assert powr(pos10001, -5, r) == 1
        assert powr(pos09999, -5, r) > 1
        assert powr(neg10001, -5, r) > -1
        assert powr(neg09999, -5, r) == -1
        assert powr(pos10001, -6, r) == 1
        assert powr(pos09999, -6, r) > 1
        assert powr(neg10001, -6, r) == 1
        assert powr(neg09999, -6, r) > 1
        r = round_floor
        assert powr(pos10001, 5, r) == 1
        assert powr(pos09999, 5, r) < 1
        assert powr(neg10001, 5, r) < -1
        assert powr(neg09999, 5, r) == -1
        assert powr(pos10001, 6, r) == 1
        assert powr(pos09999, 6, r) < 1
        assert powr(neg10001, 6, r) == 1
        assert powr(neg09999, 6, r) < 1
        assert powr(pos10001, -5, r) < 1
        assert powr(pos09999, -5, r) == 1
        assert powr(neg10001, -5, r) == -1
        assert powr(neg09999, -5, r) < -1
        assert powr(pos10001, -6, r) < 1
        assert powr(pos09999, -6, r) == 1
        assert powr(neg10001, -6, r) < 1
        assert powr(neg09999, -6, r) == 1
    mp.dps = 15