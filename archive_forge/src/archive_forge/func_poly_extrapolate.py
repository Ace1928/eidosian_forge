from copy import copy
from typing import Any, Dict, Optional, Sequence, Callable
from pennylane import apply, adjoint
from pennylane.math import mean, shape, round
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms import transform
import pennylane as qml
def poly_extrapolate(x, y, order):
    """Extrapolator to :math:`f(0)` for polynomial fit.

    The polynomial is defined as ``f(x) = p[0] * x**deg + p[1] * x**(deg-1) + ... + p[deg]`` such that ``deg = order + 1``.
    This function is compatible with all interfaces supported by pennylane.

    Args:
        x (Array): Data in x
        y (Array): Data in y = f(x)
        order (int): Order of the polynomial fit

    Returns:
        float: Extrapolated value at f(0).

    .. seealso:: :func:`~.pennylane.transforms.richardson_extrapolate`, :func:`~.pennylane.transforms.mitigate_with_zne`

    **Example:**

    >>> x = np.linspace(1, 10, 5)
    >>> y = x**2 + x + 1 + 0.3 * np.random.rand(len(x))
    >>> qml.transforms.poly_extrapolate(x, y, 2)
    tensor(1.01717601, requires_grad=True)

    """
    coeff = _polyfit(x, y, order)
    return coeff[-1]