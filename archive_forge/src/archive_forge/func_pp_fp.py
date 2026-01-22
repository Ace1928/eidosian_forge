import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_fp(self, a, d, xs):
    _z3_assert(isinstance(a, z3.FPRef), 'type mismatch')
    k = a.decl().kind()
    op = '?'
    if self.fpa_pretty and k in _z3_op_to_fpa_pretty_str:
        op = _z3_op_to_fpa_pretty_str[k]
    elif k in _z3_op_to_fpa_normal_str:
        op = _z3_op_to_fpa_normal_str[k]
    elif k in _z3_op_to_str:
        op = _z3_op_to_str[k]
    n = a.num_args()
    if self.fpa_pretty:
        if self.is_infix(k) and n >= 3:
            rm = a.arg(0)
            if z3.is_fprm_value(rm) and z3.get_default_rounding_mode(a.ctx).eq(rm):
                p = self.get_precedence(k)
                r = []
                x = a.arg(1)
                y = a.arg(2)
                arg1 = to_format(self.pp_expr(x, d + 1, xs))
                arg2 = to_format(self.pp_expr(y, d + 1, xs))
                if z3.is_app(x):
                    child_k = x.decl().kind()
                    if child_k != k and self.is_infix(child_k) and (self.get_precedence(child_k) > p):
                        arg1 = self.add_paren(arg1)
                if z3.is_app(y):
                    child_k = y.decl().kind()
                    if child_k != k and self.is_infix(child_k) and (self.get_precedence(child_k) > p):
                        arg2 = self.add_paren(arg2)
                r.append(arg1)
                r.append(to_format(' '))
                r.append(to_format(op))
                r.append(to_format(' '))
                r.append(arg2)
                return compose(r)
        elif k == Z3_OP_FPA_NEG:
            return compose([to_format('-'), to_format(self.pp_expr(a.arg(0), d + 1, xs))])
    if k in _z3_op_to_fpa_normal_str:
        op = _z3_op_to_fpa_normal_str[k]
    r = []
    r.append(to_format(op))
    if not z3.is_const(a):
        r.append(to_format('('))
        first = True
        for c in a.children():
            if first:
                first = False
            else:
                r.append(to_format(', '))
            r.append(self.pp_expr(c, d + 1, xs))
        r.append(to_format(')'))
        return compose(r)
    else:
        return to_format(a.as_string())