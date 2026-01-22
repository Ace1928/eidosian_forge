def calc_time_domain_solution(self, fp, t, manual_prec=False):
    """Calculate time-domain solution for Cohen algorithm.

        The accelerated nearly alternating series is:

        .. math ::

            f(t, M) = \\frac{e^{\\gamma / 2}}{t} \\left[\\frac{1}{2}
            \\Re\\left(\\bar{f}\\left(\\frac{\\gamma}{2t}\\right) \\right) -
            \\sum_{k=0}^{M-1}\\frac{c_{M,k}}{d_M}\\Re\\left(\\bar{f}
            \\left(\\frac{\\gamma + 2(k+1) \\pi i}{2t}\\right)\\right)\\right],

        where coefficients `\\frac{c_{M, k}}{d_M}` are described in [1].

        1. H. Cohen, F. Rodriguez Villegas, D. Zagier (2000). Convergence
        acceleration of alternating series. *Experiment. Math* 9(1):3-12

        """
    self.t = self.ctx.convert(t)
    n = self.degree
    M = n + 1
    A = self.ctx.matrix(M, 1)
    for i in range(M):
        A[i] = fp[i].real
    d = (3 + self.ctx.sqrt(8)) ** n
    d = (d + 1 / d) / 2
    b = -self.ctx.one
    c = -d
    s = 0
    for k in range(n):
        c = b - c
        s = s + c * A[k + 1]
        b = 2 * (k + n) * (k - n) * b / ((2 * k + 1) * (k + self.ctx.one))
    result = self.ctx.exp(self.alpha / 2) / self.t * (A[0] / 2 - s / d)
    if not manual_prec:
        self.ctx.dps = self.dps_orig
    return result