from __future__ import annotations
from typing import Any
from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import default_sort_key, ordered
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
def half_gcdex(self, a, b):
    """Half extended GCD of ``a`` and ``b``. """
    s, t, h = self.gcdex(a, b)
    return (s, h)