from sympy.core.sympify import sympify
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.minpoly import minpoly
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.utilities.decorator import public
from sympy.utilities.lambdify import lambdify
from mpmath import mp
def compute_up_through(self, e):
    m = self.max_so_far
    if e <= m:
        return
    n = self.n
    r = self.powers_n_and_up
    c = r[0]
    for k in range(m + 1, e + 1):
        b = r[k - 1 - n][n - 1]
        r.append([c[0] * b % self] + [(r[k - 1 - n][i - 1] + c[i] * b) % self for i in range(1, n)])
    self.max_so_far = e