from functools import reduce
import math
from operator import add
from ._expr import Expr
def eval_poly(self, variables, backend=math):
    all_args = self.all_args(variables, backend=backend)[self.skip_poly:]
    npoly = all_args[0]
    arg_idx = 1
    poly_args = []
    meta = []
    for poly_idx in range(npoly):
        meta.append(all_args[arg_idx:arg_idx + 3])
        arg_idx += 3
    for poly_idx in range(npoly):
        narg = 1 + meta[poly_idx][0]
        poly_args.append(all_args[arg_idx:arg_idx + narg])
        arg_idx += narg
    if arg_idx != len(all_args):
        raise Exception('Bug in PiecewisePoly.eval_poly')
    x = variables[parameter]
    try:
        pw = backend.Piecewise
    except AttributeError:
        for (ncoeff, lower, upper), args in zip(meta, poly_args):
            if lower <= x <= upper:
                return _eval_poly(x, args[0], args[1:], reciprocal)
        else:
            raise ValueError('not within any bounds: %s' % str(x))
    else:
        return pw(*[(_eval_poly(x, a[0], a[1:], reciprocal), backend.And(l <= x, x <= u)) for (n, l, u), a in zip(meta, poly_args)])