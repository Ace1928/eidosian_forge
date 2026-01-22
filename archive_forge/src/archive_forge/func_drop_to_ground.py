from __future__ import annotations
from typing import Any
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.numbers import igcd, oo
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import (Domain as DomainOpt,
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
def drop_to_ground(self, gen):
    if self.ring.ngens == 1:
        raise ValueError('Cannot drop only generator to ground')
    i, ring = self._drop_to_ground(gen)
    poly = ring.zero
    gen = ring.domain.gens[0]
    for monom, coeff in self.iterterms():
        mon = monom[:i] + monom[i + 1:]
        if mon not in poly:
            poly[mon] = (gen ** monom[i]).mul_ground(coeff)
        else:
            poly[mon] += (gen ** monom[i]).mul_ground(coeff)
    return poly