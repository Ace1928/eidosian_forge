from ..libmp.backend import xrange
from .calculus import defun
@defun
def fourierval(ctx, series, interval, x):
    """
    Evaluates a Fourier series (in the format computed by
    by :func:`~mpmath.fourier` for the given interval) at the point `x`.

    The series should be a pair `(c, s)` where `c` is the
    cosine series and `s` is the sine series. The two lists
    need not have the same length.
    """
    cs, ss = series
    ab = ctx._as_points(interval)
    a = interval[0]
    b = interval[-1]
    m = 2 * ctx.pi / (ab[-1] - ab[0])
    s = ctx.zero
    s += ctx.fsum((cs[n] * ctx.cos(m * n * x) for n in xrange(len(cs)) if cs[n]))
    s += ctx.fsum((ss[n] * ctx.sin(m * n * x) for n in xrange(len(ss)) if ss[n]))
    return s