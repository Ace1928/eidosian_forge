from .functions import defun, defun_wrapped
@defun
def elliprf(ctx, x, y, z):
    """
    Evaluates the Carlson symmetric elliptic integral of the first kind

    .. math ::

        R_F(x,y,z) = \\frac{1}{2}
            \\int_0^{\\infty} \\frac{dt}{\\sqrt{(t+x)(t+y)(t+z)}}

    which is defined for `x,y,z \\notin (-\\infty,0)`, and with
    at most one of `x,y,z` being zero.

    For real `x,y,z \\ge 0`, the principal square root is taken in the integrand.
    For complex `x,y,z`, the principal square root is taken as `t \\to \\infty`
    and as `t \\to 0` non-principal branches are chosen as necessary so as to
    make the integrand continuous.

    **Examples**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprf(0,1,1); pi/2
        1.570796326794896619231322
        1.570796326794896619231322
        >>> elliprf(0,1,inf)
        0.0
        >>> elliprf(1,1,1)
        1.0
        >>> elliprf(2,2,2)**2
        0.5
        >>> elliprf(1,0,0); elliprf(0,0,1); elliprf(0,1,0); elliprf(0,0,0)
        +inf
        +inf
        +inf
        +inf

    Representing complete elliptic integrals in terms of `R_F`::

        >>> m = mpf(0.75)
        >>> ellipk(m); elliprf(0,1-m,1)
        2.156515647499643235438675
        2.156515647499643235438675
        >>> ellipe(m); elliprf(0,1-m,1)-m*elliprd(0,1-m,1)/3
        1.211056027568459524803563
        1.211056027568459524803563

    Some symmetries and argument transformations::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z); elliprf(y,x,z); elliprf(z,y,x)
        0.5840828416771517066928492
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> k = mpf(100000)
        >>> elliprf(k*x,k*y,k*z); k**(-0.5) * elliprf(x,y,z)
        0.001847032121923321253219284
        0.001847032121923321253219284
        >>> l = sqrt(x*y) + sqrt(y*z) + sqrt(z*x)
        >>> elliprf(x,y,z); 2*elliprf(x+l,y+l,z+l)
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> elliprf((x+l)/4,(y+l)/4,(z+l)/4)
        0.5840828416771517066928492

    Comparing with numerical integration::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z)
        0.5840828416771517066928492
        >>> f = lambda t: 0.5*((t+x)*(t+y)*(t+z))**(-0.5)
        >>> q = extradps(25)(quad)
        >>> q(f, [0,inf])
        0.5840828416771517066928492

    With the following arguments, the square root in the integrand becomes
    discontinuous at `t = 1/2` if the principal branch is used. To obtain
    the right value, `-\\sqrt{r}` must be taken instead of `\\sqrt{r}`
    on `t \\in (0, 1/2)`::

        >>> x,y,z = j-1,j,0
        >>> elliprf(x,y,z)
        (0.7961258658423391329305694 - 1.213856669836495986430094j)
        >>> -q(f, [0,0.5]) + q(f, [0.5,inf])
        (0.7961258658423391329305694 - 1.213856669836495986430094j)

    The so-called *first lemniscate constant*, a transcendental number::

        >>> elliprf(0,1,2)
        1.31102877714605990523242
        >>> extradps(25)(quad)(lambda t: 1/sqrt(1-t**4), [0,1])
        1.31102877714605990523242
        >>> gamma('1/4')**2/(4*sqrt(2*pi))
        1.31102877714605990523242

    **References**

    1. [Carlson]_
    2. [DLMF]_ Chapter 19. Elliptic Integrals

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2 ** 10
        v = RF_calc(ctx, x, y, z, tol)
    finally:
        ctx.prec = prec
    return +v