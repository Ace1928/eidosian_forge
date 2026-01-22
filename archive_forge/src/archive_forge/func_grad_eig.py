from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    e, u = ans
    n = e.shape[-1]

    def vjp(g):
        ge, gu = g
        ge = _matrix_diag(ge)
        f = 1 / (e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * _dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut, gu)) * anp.eye(n))
        r = _dot(_dot(inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
        return r
    return vjp