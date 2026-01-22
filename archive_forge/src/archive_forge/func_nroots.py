from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import (
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import (
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.numbers import ilcm, I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (
from sympy.polys.polyutils import (
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence
def nroots(f, n=15, maxsteps=50, cleanup=True):
    """
        Compute numerical approximations of roots of ``f``.

        Parameters
        ==========

        n ... the number of digits to calculate
        maxsteps ... the maximum number of iterations to do

        If the accuracy `n` cannot be reached in `maxsteps`, it will raise an
        exception. You need to rerun with higher maxsteps.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x

        >>> Poly(x**2 - 3).nroots(n=15)
        [-1.73205080756888, 1.73205080756888]
        >>> Poly(x**2 - 3).nroots(n=30)
        [-1.73205080756887729352744634151, 1.73205080756887729352744634151]

        """
    if f.is_multivariate:
        raise MultivariatePolynomialError('Cannot compute numerical roots of %s' % f)
    if f.degree() <= 0:
        return []
    if f.rep.dom is ZZ:
        coeffs = [int(coeff) for coeff in f.all_coeffs()]
    elif f.rep.dom is QQ:
        denoms = [coeff.q for coeff in f.all_coeffs()]
        fac = ilcm(*denoms)
        coeffs = [int(coeff * fac) for coeff in f.all_coeffs()]
    else:
        coeffs = [coeff.evalf(n=n).as_real_imag() for coeff in f.all_coeffs()]
        try:
            coeffs = [mpmath.mpc(*coeff) for coeff in coeffs]
        except TypeError:
            raise DomainError('Numerical domain expected, got %s' % f.rep.dom)
    dps = mpmath.mp.dps
    mpmath.mp.dps = n
    from sympy.functions.elementary.complexes import sign
    try:
        roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 10)
        roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
    except NoConvergence:
        try:
            roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 15)
            roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
        except NoConvergence:
            raise NoConvergence('convergence to root failed; try n < %s or maxsteps > %s' % (n, maxsteps))
    finally:
        mpmath.mp.dps = dps
    return roots