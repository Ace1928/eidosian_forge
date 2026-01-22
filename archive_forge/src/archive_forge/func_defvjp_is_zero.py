from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
def defvjp_is_zero(argnums=(0,)):
    warnings.warn(deprecation_msg)
    zero_vjps[0] |= set(argnums)
    nones = [None] * len(zero_vjps[0])
    defvjp(primitive_fun, *nones, argnums=sorted(zero_vjps[0]))