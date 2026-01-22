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
def eject(f, *gens):
    """
        Eject selected generators into the ground domain.

        Examples
        ========

        >>> from sympy import Poly
        >>> from sympy.abc import x, y

        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

        >>> f.eject(x)
        Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
        >>> f.eject(y)
        Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

        """
    dom = f.rep.dom
    if not dom.is_Numerical:
        raise DomainError('Cannot eject generators over %s' % dom)
    k = len(gens)
    if f.gens[:k] == gens:
        _gens, front = (f.gens[k:], True)
    elif f.gens[-k:] == gens:
        _gens, front = (f.gens[:-k], False)
    else:
        raise NotImplementedError('can only eject front or back generators')
    dom = dom.inject(*gens)
    if hasattr(f.rep, 'eject'):
        result = f.rep.eject(dom, front=front)
    else:
        raise OperationNotSupported(f, 'eject')
    return f.new(result, *_gens)