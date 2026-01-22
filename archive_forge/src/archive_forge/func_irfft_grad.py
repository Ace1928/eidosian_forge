from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def irfft_grad(get_args, rfft_fun, ans, x, *args, **kwargs):
    axes, gs, norm = get_args(x, *args, **kwargs)
    vs = vspace(x)
    gvs = vspace(ans)
    check_no_repeated_axes(axes)
    if gs is None:
        gs = [gvs.shape[i] for i in axes]
    check_even_shape(gs)
    s = list(gs)
    s[-1] = s[-1] // 2 + 1

    def vjp(g):
        r = match_complex(x, truncate_pad(rfft_fun(g, *args, **kwargs), vs.shape))
        fac = make_rfft_factors(axes, vs.shape, s, gs, norm)
        r = anp.conj(r) * fac
        return r
    return vjp