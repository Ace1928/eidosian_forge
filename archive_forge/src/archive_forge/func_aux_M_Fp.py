import math
from .functions import defun
def aux_M_Fp(ctx, xA, xeps4, a, xB1, xL):
    aux1 = 126.0657606 * xA / xeps4
    aux1 = ctx.ln(aux1)
    aux2 = (2 * ctx.ln(ctx.pi) + ctx.ln(xB1) + ctx.ln(a)) / 3 - ctx.ln(2 * ctx.pi) / 2
    m = 3 * xL - 3
    aux3 = (ctx.loggamma(m + 1) - ctx.loggamma(m / 3.0 + 2)) / 2 - ctx.loggamma((m + 1) / 2.0)
    while aux1 < m * aux2 + aux3 and m > 1:
        m = m - 1
        aux3 = (ctx.loggamma(m + 1) - ctx.loggamma(m / 3.0 + 2)) / 2 - ctx.loggamma((m + 1) / 2.0)
    xM = m
    return xM