import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
def grad_sequence_extend_left(argnum, ans, args, kwargs):
    seq, elts = (args[0], args[1:])
    return lambda g: g[len(elts):] if argnum == 0 else g[argnum - 1]