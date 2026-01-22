from math import exp, sin, cos
def salustowicz_1d(data):
    """Salustowicz benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`x \\in [0, 10]`
       * - Function
         - :math:`f(x) = e^{-x} x^3 \\cos(x) \\sin(x) (\\cos(x) \\sin^2(x) - 1)`
    """
    return exp(-data[0]) * data[0] ** 3 * cos(data[0]) * sin(data[0]) * (cos(data[0]) * sin(data[0]) ** 2 - 1)