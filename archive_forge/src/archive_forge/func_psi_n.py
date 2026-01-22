from sympy.core import S, pi, Rational
from sympy.functions import hermite, sqrt, exp, factorial, Abs
from sympy.physics.quantum.constants import hbar
def psi_n(n, x, m, omega):
    """
    Returns the wavefunction psi_{n} for the One-dimensional harmonic oscillator.

    Parameters
    ==========

    n :
        the "nodal" quantum number.  Corresponds to the number of nodes in the
        wavefunction.  ``n >= 0``
    x :
        x coordinate.
    m :
        Mass of the particle.
    omega :
        Angular frequency of the oscillator.

    Examples
    ========

    >>> from sympy.physics.qho_1d import psi_n
    >>> from sympy.abc import m, x, omega
    >>> psi_n(0, x, m, omega)
    (m*omega)**(1/4)*exp(-m*omega*x**2/(2*hbar))/(hbar**(1/4)*pi**(1/4))

    """
    n, x, m, omega = map(S, [n, x, m, omega])
    nu = m * omega / hbar
    C = (nu / pi) ** Rational(1, 4) * sqrt(1 / (2 ** n * factorial(n)))
    return C * exp(-nu * x ** 2 / 2) * hermite(n, sqrt(nu) * x)