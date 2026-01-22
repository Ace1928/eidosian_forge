import math
from .functions import defun
def coef(ctx, J, eps):
    _cache = ctx._rs_cache
    if J <= _cache[0] and eps >= _cache[1]:
        return (_cache[2], _cache[3])
    orig = ctx._mp.prec
    try:
        data = _coef(ctx._mp, J, eps)
    finally:
        ctx._mp.prec = orig
    if ctx is not ctx._mp:
        data[2] = dict(((k, ctx.convert(v)) for k, v in data[2].items()))
        data[3] = dict(((k, ctx.convert(v)) for k, v in data[3].items()))
    ctx._rs_cache[:] = data
    return (ctx._rs_cache[2], ctx._rs_cache[3])