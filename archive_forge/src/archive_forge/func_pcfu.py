from .functions import defun, defun_wrapped
@defun
def pcfu(ctx, a, z, **kwargs):
    """
    Gives the parabolic cylinder function `U(a,z)`, which may be
    defined for `\\Re(z) > 0` in terms of the confluent
    U-function (see :func:`~mpmath.hyperu`) by

    .. math ::

        U(a,z) = 2^{-\\frac{1}{4}-\\frac{a}{2}} e^{-\\frac{1}{4} z^2}
            U\\left(\\frac{a}{2}+\\frac{1}{4},
            \\frac{1}{2}, \\frac{1}{2}z^2\\right)

    or, for arbitrary `z`,

    .. math ::

        e^{-\\frac{1}{4}z^2} U(a,z) =
            U(a,0) \\,_1F_1\\left(-\\tfrac{a}{2}+\\tfrac{1}{4};
            \\tfrac{1}{2}; -\\tfrac{1}{2}z^2\\right) +
            U'(a,0) z \\,_1F_1\\left(-\\tfrac{a}{2}+\\tfrac{3}{4};
            \\tfrac{3}{2}; -\\tfrac{1}{2}z^2\\right).

    **Examples**

    Connection to other functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> z = mpf(3)
        >>> pcfu(0.5,z)
        0.03210358129311151450551963
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(z/sqrt(2))
        0.03210358129311151450551963
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255

    """
    n, _ = ctx._convert_param(a)
    return ctx.pcfd(-n - ctx.mpq_1_2, z)