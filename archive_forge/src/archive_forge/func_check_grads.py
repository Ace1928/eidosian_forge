from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name
@unary_to_nary
def check_grads(f, x, modes=['fwd', 'rev'], order=2):
    assert all((m in ['fwd', 'rev'] for m in modes))
    if 'fwd' in modes:
        check_jvp(f, x)
        if order > 1:
            grad_f = lambda x, v: make_jvp(f, x)(v)[1]
            grad_f.__name__ = 'jvp_{}'.format(get_name(f))
            v = vspace(x).randn()
            check_grads(grad_f, (0, 1), modes, order=order - 1)(x, v)
    if 'rev' in modes:
        check_vjp(f, x)
        if order > 1:
            grad_f = lambda x, v: make_vjp(f, x)[0](v)
            grad_f.__name__ = 'vjp_{}'.format(get_name(f))
            v = vspace(f(x)).randn()
            check_grads(grad_f, (0, 1), modes, order=order - 1)(x, v)