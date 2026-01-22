from ...sage_helper import _within_sage, sage_method
@sage_method
def my_dilog(z):
    """
    Compute dilogarithm using complex ball field.
    The dilogarithm isn't implemented for ComplexIntervalField itself, so
    we use ComplexBallField. Note that ComplexBallField is conservative
    about branch cuts. For Li_2(2+-i * epsilon), it returns the interval
    containing both Li_2(2+i * epsilon) and Li_2(2-i * epsilon).

    Thus, we need to avoid calling this function with a value near real numbers
    greater 1.
    """
    CIF = z.parent()
    CBF = ComplexBallField(CIF.precision())
    return CIF(CBF(z).polylog(2))