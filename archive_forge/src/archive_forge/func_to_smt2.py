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
def to_smt2(self):
    """return SMTLIB2 formatted benchmark for solver's assertions"""
    es = self.assertions()
    sz = len(es)
    sz1 = sz
    if sz1 > 0:
        sz1 -= 1
    v = (Ast * sz1)()
    for i in range(sz1):
        v[i] = es[i].as_ast()
    if sz > 0:
        e = es[sz1].as_ast()
    else:
        e = BoolVal(True, self.ctx).as_ast()
    return Z3_benchmark_to_smtlib_string(self.ctx.ref(), 'benchmark generated from python API', '', 'unknown', '', sz1, v, e)