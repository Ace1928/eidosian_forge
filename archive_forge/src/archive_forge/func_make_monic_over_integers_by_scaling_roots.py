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
def make_monic_over_integers_by_scaling_roots(f):
    """
        Turn any univariate polynomial over :ref:`QQ` or :ref:`ZZ` into a monic
        polynomial over :ref:`ZZ`, by scaling the roots as necessary.

        Explanation
        ===========

        This operation can be performed whether or not *f* is irreducible; when
        it is, this can be understood as determining an algebraic integer
        generating the same field as a root of *f*.

        Examples
        ========

        >>> from sympy import Poly, S
        >>> from sympy.abc import x
        >>> f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
        >>> f.make_monic_over_integers_by_scaling_roots()
        (Poly(x**2 + 2*x + 4, x, domain='ZZ'), 4)

        Returns
        =======

        Pair ``(g, c)``
            g is the polynomial

            c is the integer by which the roots had to be scaled

        """
    if not f.is_univariate or f.domain not in [ZZ, QQ]:
        raise ValueError('Polynomial must be univariate over ZZ or QQ.')
    if f.is_monic and f.domain == ZZ:
        return (f, ZZ.one)
    else:
        fm = f.monic()
        c, _ = fm.clear_denoms()
        return (fm.transform(Poly(fm.gen), c).to_ring(), c)