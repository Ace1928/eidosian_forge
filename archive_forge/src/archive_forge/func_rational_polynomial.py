from math import exp, sin, cos
def rational_polynomial(data):
    """Rational polynomial ball benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [0, 2]^3`
       * - Function
         - :math:`f(\\mathbf{x}) = \\\\frac{30 * (x_1 - 1) (x_3 - 1)}{x_2^2 (x_1 - 10)}`
    """
    return 30.0 * (data[0] - 1) * (data[2] - 1) / (data[1] ** 2 * (data[0] - 10))