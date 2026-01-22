from .backend import xrange
from .libmpf import (
from .libelefun import (
from .gammazeta import mpf_gamma, mpf_rgamma, mpf_loggamma, mpc_loggamma
def mpi_to_str(x, dps, use_spaces=True, brackets='[]', mode='brackets', error_dps=4, **kwargs):
    """
    Convert a mpi interval to a string.

    **Arguments**

    *dps*
        decimal places to use for printing
    *use_spaces*
        use spaces for more readable output, defaults to true
    *brackets*
        pair of strings (or two-character string) giving left and right brackets
    *mode*
        mode of display: 'plusminus', 'percent', 'brackets' (default) or 'diff'
    *error_dps*
        limit the error to *error_dps* digits (mode 'plusminus and 'percent')

    Additional keyword arguments are forwarded to the mpf-to-string conversion
    for the components of the output.

    **Examples**

        >>> from mpmath import mpi, mp
        >>> mp.dps = 30
        >>> x = mpi(1, 2)._mpi_
        >>> mpi_to_str(x, 2, mode='plusminus')
        '1.5 +- 0.5'
        >>> mpi_to_str(x, 2, mode='percent')
        '1.5 (33.33%)'
        >>> mpi_to_str(x, 2, mode='brackets')
        '[1.0, 2.0]'
        >>> mpi_to_str(x, 2, mode='brackets' , brackets=('<', '>'))
        '<1.0, 2.0>'
        >>> x = mpi('5.2582327113062393041', '5.2582327113062749951')._mpi_
        >>> mpi_to_str(x, 15, mode='diff')
        '5.2582327113062[4, 7]'
        >>> mpi_to_str(mpi(0)._mpi_, 2, mode='percent')
        '0.0 (0.0%)'

    """
    prec = dps_to_prec(dps)
    wp = prec + 20
    a, b = x
    mid = mpi_mid(x, prec)
    delta = mpi_delta(x, prec)
    a_str = to_str(a, dps, **kwargs)
    b_str = to_str(b, dps, **kwargs)
    mid_str = to_str(mid, dps, **kwargs)
    sp = ''
    if use_spaces:
        sp = ' '
    br1, br2 = brackets
    if mode == 'plusminus':
        delta_str = to_str(mpf_shift(delta, -1), dps, **kwargs)
        s = mid_str + sp + '+-' + sp + delta_str
    elif mode == 'percent':
        if mid == fzero:
            p = fzero
        else:
            p = mpf_mul(delta, from_int(100))
            p = mpf_div(p, mpf_mul(mid, from_int(2)), wp)
        s = mid_str + sp + '(' + to_str(p, error_dps) + '%)'
    elif mode == 'brackets':
        s = br1 + a_str + ',' + sp + b_str + br2
    elif mode == 'diff':
        if a_str == b_str:
            a_str = to_str(a, dps + 3, **kwargs)
            b_str = to_str(b, dps + 3, **kwargs)
        a = a_str.split('e')
        if len(a) == 1:
            a.append('')
        b = b_str.split('e')
        if len(b) == 1:
            b.append('')
        if a[1] == b[1]:
            if a[0] != b[0]:
                for i in xrange(len(a[0]) + 1):
                    if a[0][i] != b[0][i]:
                        break
                s = a[0][:i] + br1 + a[0][i:] + ',' + sp + b[0][i:] + br2 + 'e' * min(len(a[1]), 1) + a[1]
            else:
                s = a[0] + br1 + br2 + 'e' * min(len(a[1]), 1) + a[1]
        else:
            s = br1 + 'e'.join(a) + ',' + sp + 'e'.join(b) + br2
    else:
        raise ValueError("'%s' is unknown mode for printing mpi" % mode)
    return s