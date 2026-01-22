from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def parse_smt2_string(s, sorts={}, decls={}, ctx=None):
    """Parse a string in SMT 2.0 format using the given sorts and decls.

    The arguments sorts and decls are Python dictionaries used to initialize
    the symbol table used for the SMT 2.0 parser.

    >>> parse_smt2_string('(declare-const x Int) (assert (> x 0)) (assert (< x 10))')
    [x > 0, x < 10]
    >>> x, y = Ints('x y')
    >>> f = Function('f', IntSort(), IntSort())
    >>> parse_smt2_string('(assert (> (+ foo (g bar)) 0))', decls={ 'foo' : x, 'bar' : y, 'g' : f})
    [x + f(y) > 0]
    >>> parse_smt2_string('(declare-const a U) (assert (> a 0))', sorts={ 'U' : IntSort() })
    [a > 0]
    """
    ctx = _get_ctx(ctx)
    ssz, snames, ssorts = _dict2sarray(sorts, ctx)
    dsz, dnames, ddecls = _dict2darray(decls, ctx)
    return AstVector(Z3_parse_smtlib2_string(ctx.ref(), s, ssz, snames, ssorts, dsz, dnames, ddecls), ctx)