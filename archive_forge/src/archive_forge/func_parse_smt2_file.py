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
def parse_smt2_file(f, sorts={}, decls={}, ctx=None):
    """Parse a file in SMT 2.0 format using the given sorts and decls.

    This function is similar to parse_smt2_string().
    """
    ctx = _get_ctx(ctx)
    ssz, snames, ssorts = _dict2sarray(sorts, ctx)
    dsz, dnames, ddecls = _dict2darray(decls, ctx)
    return AstVector(Z3_parse_smtlib2_file(ctx.ref(), f, ssz, snames, ssorts, dsz, dnames, ddecls), ctx)