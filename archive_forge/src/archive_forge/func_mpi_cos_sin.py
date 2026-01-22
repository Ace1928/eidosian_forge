from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_cos_sin(x, prec):
    a, b = x
    if a == b == fzero:
        return ((fone, fone), (fzero, fzero))
    if finf in x or fninf in x:
        return ((fnone, fone), (fnone, fone))
    wp = prec + 20
    ca, sa, na = cos_sin_quadrant(a, wp)
    cb, sb, nb = cos_sin_quadrant(b, wp)
    ca, cb = mpf_min_max([ca, cb])
    sa, sb = mpf_min_max([sa, sb])
    if na == nb:
        pass
    elif nb - na >= 4:
        return ((fnone, fone), (fnone, fone))
    else:
        if na // 4 != nb // 4:
            cb = fone
        if (na - 2) // 4 != (nb - 2) // 4:
            ca = fnone
        if (na - 1) // 4 != (nb - 1) // 4:
            sb = fone
        if (na - 3) // 4 != (nb - 3) // 4:
            sa = fnone
    more = from_man_exp((MPZ_ONE << wp) + (MPZ_ONE << 10), -wp)
    less = from_man_exp((MPZ_ONE << wp) - (MPZ_ONE << 10), -wp)

    def finalize(v, rounding):
        if bool(v[0]) == (rounding == round_floor):
            p = more
        else:
            p = less
        v = mpf_mul(v, p, prec, rounding)
        sign, man, exp, bc = v
        if exp + bc >= 1:
            if sign:
                return fnone
            return fone
        return v
    ca = finalize(ca, round_floor)
    cb = finalize(cb, round_ceiling)
    sa = finalize(sa, round_floor)
    sb = finalize(sb, round_ceiling)
    return ((ca, cb), (sa, sb))