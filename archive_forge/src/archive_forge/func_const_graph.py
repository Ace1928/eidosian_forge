from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
def const_graph(fun, *args, **kwargs):
    partial_fun = partial(fun, *args, **kwargs)
    unary_fun = lambda args: partial_fun(*args)
    maybe_cached_unary_fun = const_graph_unary(unary_fun)

    @wraps(fun)
    def _fun(*args):
        return maybe_cached_unary_fun(args)
    return _fun