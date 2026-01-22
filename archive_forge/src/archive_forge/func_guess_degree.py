import math
from ..libmp.backend import xrange
def guess_degree(self, prec):
    """
        Given a desired precision `p` in bits, estimate the degree `m`
        of the quadrature required to accomplish full accuracy for
        typical integrals. By default, :func:`~mpmath.quad` will perform up
        to `m` iterations. The value of `m` should be a slight
        overestimate, so that "slightly bad" integrals can be dealt
        with automatically using a few extra iterations. On the
        other hand, it should not be too big, so :func:`~mpmath.quad` can
        quit within a reasonable amount of time when it is given
        an "unsolvable" integral.

        The default formula used by :func:`~mpmath.guess_degree` is tuned
        for both :class:`TanhSinh` and :class:`GaussLegendre`.
        The output is roughly as follows:

            +---------+---------+
            | `p`     | `m`     |
            +=========+=========+
            | 50      | 6       |
            +---------+---------+
            | 100     | 7       |
            +---------+---------+
            | 500     | 10      |
            +---------+---------+
            | 3000    | 12      |
            +---------+---------+

        This formula is based purely on a limited amount of
        experimentation and will sometimes be wrong.
        """
    g = int(4 + max(0, self.ctx.log(prec / 30.0, 2)))
    g += 2
    return g