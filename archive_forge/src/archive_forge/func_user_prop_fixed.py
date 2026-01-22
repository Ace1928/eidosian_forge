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
def user_prop_fixed(ctx, cb, id, value):
    prop = _prop_closures.get(ctx)
    old_cb = prop.cb
    prop.cb = cb
    id = _to_expr_ref(to_Ast(id), prop.ctx())
    value = _to_expr_ref(to_Ast(value), prop.ctx())
    prop.fixed(id, value)
    prop.cb = old_cb