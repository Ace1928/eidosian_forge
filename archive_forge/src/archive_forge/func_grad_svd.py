from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def grad_svd(usv_, a, full_matrices=True, compute_uv=True):

    def vjp(g):
        usv = usv_
        if not compute_uv:
            s = usv
            usv = svd(a, full_matrices=False)
            u = usv[0]
            v = anp.conj(T(usv[2]))
            return _dot(anp.conj(u) * g[..., anp.newaxis, :], T(v))
        elif full_matrices:
            raise NotImplementedError('Gradient of svd not implemented for full_matrices=True')
        else:
            u = usv[0]
            s = usv[1]
            v = anp.conj(T(usv[2]))
            m, n = a.shape[-2:]
            k = anp.min((m, n))
            i = anp.reshape(anp.eye(k), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (k, k))))
            f = 1 / (s[..., anp.newaxis, :] ** 2 - s[..., :, anp.newaxis] ** 2 + i)
            gu = g[0]
            gs = g[1]
            gv = anp.conj(T(g[2]))
            utgu = _dot(T(u), gu)
            vtgv = _dot(T(v), gv)
            t1 = f * (utgu - anp.conj(T(utgu))) * s[..., anp.newaxis, :]
            t1 = t1 + i * gs[..., :, anp.newaxis]
            t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - anp.conj(T(vtgv))))
            if anp.iscomplexobj(u):
                t1 = t1 + 1j * anp.imag(_diag(utgu)) / s[..., anp.newaxis, :]
            t1 = _dot(_dot(anp.conj(u), t1), T(v))
            if m < n:
                i_minus_vvt = anp.reshape(anp.eye(n), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (n, n)))) - _dot(v, anp.conj(T(v)))
                t1 = t1 + anp.conj(_dot(_dot(u / s[..., anp.newaxis, :], T(gv)), i_minus_vvt))
                return t1
            elif m == n:
                return t1
            elif m > n:
                i_minus_uut = anp.reshape(anp.eye(m), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (m, m)))) - _dot(u, anp.conj(T(u)))
                t1 = t1 + T(_dot(_dot(v / s[..., anp.newaxis, :], T(gu)), i_minus_uut))
                return t1
    return vjp