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
def count_roots(f, inf=None, sup=None):
    """
        Return the number of roots of ``f`` in ``[inf, sup]`` interval.

        Examples
        ========

        >>> from sympy import Poly, I
        >>> from sympy.abc import x

        >>> Poly(x**4 - 4, x).count_roots(-3, 3)
        2
        >>> Poly(x**4 - 4, x).count_roots(0, 1 + 3*I)
        1

        """
    inf_real, sup_real = (True, True)
    if inf is not None:
        inf = sympify(inf)
        if inf is S.NegativeInfinity:
            inf = None
        else:
            re, im = inf.as_real_imag()
            if not im:
                inf = QQ.convert(inf)
            else:
                inf, inf_real = (list(map(QQ.convert, (re, im))), False)
    if sup is not None:
        sup = sympify(sup)
        if sup is S.Infinity:
            sup = None
        else:
            re, im = sup.as_real_imag()
            if not im:
                sup = QQ.convert(sup)
            else:
                sup, sup_real = (list(map(QQ.convert, (re, im))), False)
    if inf_real and sup_real:
        if hasattr(f.rep, 'count_real_roots'):
            count = f.rep.count_real_roots(inf=inf, sup=sup)
        else:
            raise OperationNotSupported(f, 'count_real_roots')
    else:
        if inf_real and inf is not None:
            inf = (inf, QQ.zero)
        if sup_real and sup is not None:
            sup = (sup, QQ.zero)
        if hasattr(f.rep, 'count_complex_roots'):
            count = f.rep.count_complex_roots(inf=inf, sup=sup)
        else:
            raise OperationNotSupported(f, 'count_complex_roots')
    return Integer(count)