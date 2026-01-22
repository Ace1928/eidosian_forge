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
def deflate(f, *G):
    ring = f.ring
    polys = [f] + list(G)
    J = [0] * ring.ngens
    for p in polys:
        for monom in p.itermonoms():
            for i, m in enumerate(monom):
                J[i] = igcd(J[i], m)
    for i, b in enumerate(J):
        if not b:
            J[i] = 1
    J = tuple(J)
    if all((b == 1 for b in J)):
        return (J, polys)
    H = []
    for p in polys:
        h = ring.zero
        for I, coeff in p.iterterms():
            N = [i // j for i, j in zip(I, J)]
            h[tuple(N)] = coeff
        H.append(h)
    return (J, H)