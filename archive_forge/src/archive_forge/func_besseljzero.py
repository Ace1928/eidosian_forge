from .functions import defun, defun_wrapped
@defun
def besseljzero(ctx, v, m, derivative=0):
    """
    For a real order `\\nu \\ge 0` and a positive integer `m`, returns
    `j_{\\nu,m}`, the `m`-th positive zero of the Bessel function of the
    first kind `J_{\\nu}(z)` (see :func:`~mpmath.besselj`). Alternatively,
    with *derivative=1*, gives the first nonnegative simple zero
    `j'_{\\nu,m}` of `J'_{\\nu}(z)`.

    The indexing convention is that used by Abramowitz & Stegun
    and the DLMF. Note the special case `j'_{0,1} = 0`, while all other
    zeros are positive. In effect, only simple zeros are counted
    (all zeros of Bessel functions are simple except possibly `z = 0`)
    and `j_{\\nu,m}` becomes a monotonic function of both `\\nu`
    and `m`.

    The zeros are interlaced according to the inequalities

    .. math ::

        j'_{\\nu,k} < j_{\\nu,k} < j'_{\\nu,k+1}

        j_{\\nu,1} < j_{\\nu+1,2} < j_{\\nu,2} < j_{\\nu+1,2} < j_{\\nu,3} < \\cdots

    **Examples**

    Initial zeros of the Bessel functions `J_0(z), J_1(z), J_2(z)`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> besseljzero(0,1); besseljzero(0,2); besseljzero(0,3)
        2.404825557695772768621632
        5.520078110286310649596604
        8.653727912911012216954199
        >>> besseljzero(1,1); besseljzero(1,2); besseljzero(1,3)
        3.831705970207512315614436
        7.01558666981561875353705
        10.17346813506272207718571
        >>> besseljzero(2,1); besseljzero(2,2); besseljzero(2,3)
        5.135622301840682556301402
        8.417244140399864857783614
        11.61984117214905942709415

    Initial zeros of `J'_0(z), J'_1(z), J'_2(z)`::

        0.0
        3.831705970207512315614436
        7.01558666981561875353705
        >>> besseljzero(1,1,1); besseljzero(1,2,1); besseljzero(1,3,1)
        1.84118378134065930264363
        5.331442773525032636884016
        8.536316366346285834358961
        >>> besseljzero(2,1,1); besseljzero(2,2,1); besseljzero(2,3,1)
        3.054236928227140322755932
        6.706133194158459146634394
        9.969467823087595793179143

    Zeros with large index::

        >>> besseljzero(0,100); besseljzero(0,1000); besseljzero(0,10000)
        313.3742660775278447196902
        3140.807295225078628895545
        31415.14114171350798533666
        >>> besseljzero(5,100); besseljzero(5,1000); besseljzero(5,10000)
        321.1893195676003157339222
        3148.657306813047523500494
        31422.9947255486291798943
        >>> besseljzero(0,100,1); besseljzero(0,1000,1); besseljzero(0,10000,1)
        311.8018681873704508125112
        3139.236339643802482833973
        31413.57032947022399485808

    Zeros of functions with large order::

        >>> besseljzero(50,1)
        57.11689916011917411936228
        >>> besseljzero(50,2)
        62.80769876483536093435393
        >>> besseljzero(50,100)
        388.6936600656058834640981
        >>> besseljzero(50,1,1)
        52.99764038731665010944037
        >>> besseljzero(50,2,1)
        60.02631933279942589882363
        >>> besseljzero(50,100,1)
        387.1083151608726181086283

    Zeros of functions with fractional order::

        >>> besseljzero(0.5,1); besseljzero(1.5,1); besseljzero(2.25,4)
        3.141592653589793238462643
        4.493409457909064175307881
        15.15657692957458622921634

    Both `J_{\\nu}(z)` and `J'_{\\nu}(z)` can be expressed as infinite
    products over their zeros::

        >>> v,z = 2, mpf(1)
        >>> (z/2)**v/gamma(v+1) * \\
        ...     nprod(lambda k: 1-(z/besseljzero(v,k))**2, [1,inf])
        ...
        0.1149034849319004804696469
        >>> besselj(v,z)
        0.1149034849319004804696469
        >>> (z/2)**(v-1)/2/gamma(v) * \\
        ...     nprod(lambda k: 1-(z/besseljzero(v,k,1))**2, [1,inf])
        ...
        0.2102436158811325550203884
        >>> besselj(v,z,1)
        0.2102436158811325550203884

    """
    return +bessel_zero(ctx, 1, derivative, v, m)