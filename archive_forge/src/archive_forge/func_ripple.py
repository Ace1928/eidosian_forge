from math import exp, sin, cos
def ripple(data):
    """Ripple benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\\mathbf{x} \\in [-5, 5]^2`
       * - Function
         - :math:`f(\\mathbf{x}) = (x_1 - 3) (x_2 - 3) + 2 \\sin((x_1 - 4) (x_2 -4))`
    """
    return (data[0] - 3) * (data[1] - 3) + 2 * sin((data[0] - 4) * (data[1] - 4))