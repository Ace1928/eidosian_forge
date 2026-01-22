from .functions import defun_wrapped
@defun_wrapped
def trianglew(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period
    return 2 * A * (0.5 - ctx.fabs(1 - 2 * ctx.frac(t / P + 0.25)))