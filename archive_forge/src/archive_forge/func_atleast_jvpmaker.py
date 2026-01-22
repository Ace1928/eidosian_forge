from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
from ..util import func
from .numpy_boxes import ArrayBox
def atleast_jvpmaker(fun):

    def jvp(g, ans, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return fun(g)
    return jvp