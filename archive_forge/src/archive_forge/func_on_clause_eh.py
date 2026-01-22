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
def on_clause_eh(ctx, p, n, dep, clause):
    onc = _my_hacky_class
    p = _to_expr_ref(to_Ast(p), onc.ctx)
    clause = AstVector(to_AstVectorObj(clause), onc.ctx)
    deps = [dep[i] for i in range(n)]
    onc.on_clause(p, deps, clause)