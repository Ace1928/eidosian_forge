from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def defjvp(fun, *jvpfuns, **kwargs):
    argnums = kwargs.get('argnums', count())
    jvps_dict = {argnum: translate_jvp(jvpfun, fun, argnum) for argnum, jvpfun in zip(argnums, jvpfuns)}

    def jvp_argnums(argnums, gs, ans, args, kwargs):
        return sum_outgrads((jvps_dict[argnum](g, ans, *args, **kwargs) for argnum, g in zip(argnums, gs)))
    defjvp_argnums(fun, jvp_argnums)