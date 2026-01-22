from typing import Sequence, Callable
import functools
from functools import partial
from warnings import warn
import numpy as np
from scipy.special import factorial
from scipy.linalg import solve as linalg_solve
import pennylane as qml
from pennylane.measurements import ProbabilityMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .general_shift_rules import generate_shifted_tapes
from .gradient_transform import (
@functools.lru_cache(maxsize=None)
def finite_diff_coeffs(n, approx_order, strategy):
    """Generate the finite difference shift values and corresponding
    term coefficients for a given derivative order, approximation accuracy,
    and strategy.

    Args:
        n (int): Positive integer specifying the order of the derivative. For example, ``n=1``
            corresponds to the first derivative, ``n=2`` the second derivative, etc.
        approx_order (int): Positive integer referring to the approximation order of the
            returned coefficients, e.g., ``approx_order=1`` corresponds to the
            first-order approximation to the derivative.
        strategy (str): One of ``"forward"``, ``"center"``, or ``"backward"``.
            For the ``"forward"`` strategy, the finite-difference shifts occur at the points
            :math:`x_0, x_0+h, x_0+2h,\\dots`, where :math:`h` is some small
            step size. The ``"backwards"`` strategy is similar, but in
            reverse: :math:`x_0, x_0-h, x_0-2h, \\dots`. Finally, the
            ``"center"`` strategy results in shifts symmetric around the
            unshifted point: :math:`\\dots, x_0-2h, x_0-h, x_0, x_0+h, x_0+2h,\\dots`.

    Returns:
        array[float]: A ``(2, N)`` array. The first row corresponds to the
        coefficients, and the second row corresponds to the shifts.

    **Example**

    >>> finite_diff_coeffs(n=1, approx_order=1, strategy="forward")
    array([[-1.,  1.],
           [ 0.,  1.]])

    For example, this results in the linear combination:

    .. math:: \\frac{-y(x_0) + y(x_0 + h)}{h}

    where :math:`h` is the finite-difference step size.

    More examples:

    >>> finite_diff_coeffs(n=1, approx_order=2, strategy="center")
    array([[-0.5,  0.5],
           [-1. ,  1. ]])
    >>> finite_diff_coeffs(n=2, approx_order=2, strategy="center")
    array([[-2.,  1.,  1.],
           [ 0., -1.,  1.]])

    **Details**

    Consider a function :math:`y(x)`. We wish to approximate the :math:`n`-th
    derivative at point :math:`x_0`, :math:`y^{(n)}(x_0)`, by sampling the function
    at :math:`N<n` distinct points:

    .. math:: y^{(n)}(x_0) \\approx \\sum_{i=1}^N c_i y(x_i)

    where :math:`c_i` are coefficients, and :math:`x_i=x_0 + s_i` are the points we sample
    the function at.

    Consider the Taylor expansion of :math:`y(x_i)` around the point :math:`x_0`:

    .. math::

        y^{(n)}(x_0) \\approx \\sum_{i=1}^N c_i y(x_i)
            &= \\sum_{i=1}^N c_i \\left[ y(x_0) + y'(x_0)(x_i-x_0) + \\frac{1}{2} y''(x_0)(x_i-x_0)^2 + \\cdots \\right]\\\\
            & = \\sum_{j=0}^m y^{(j)}(x_0) \\left[\\sum_{i=1}^N \\frac{c_i s_i^j}{j!} + \\mathcal{O}(s_i^m) \\right],

    where :math:`s_i = x_i-x_0`. For this approximation to be satisfied, we must therefore have

    .. math::

        \\sum_{i=1}^N s_i^j c_i = \\begin{cases} j!, &j=n\\\\ 0, & j\\neq n\\end{cases}.

    Thus, to determine the coefficients :math:`c_i \\in \\{c_1, \\dots, c_N\\}` for particular
    shift values :math:`s_i \\in \\{s_1, \\dots, s_N\\}` and derivative order :math:`n`,
    we must solve this linear system of equations.
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError('Derivative order n must be a positive integer.')
    if approx_order < 1 or not isinstance(approx_order, int):
        raise ValueError('Approximation order must be a positive integer.')
    num_points = approx_order + 2 * np.floor((n + 1) / 2) - 1
    N = num_points + 1 if n % 2 == 0 else num_points
    if strategy == 'forward':
        shifts = np.arange(N, dtype=np.float64)
    elif strategy == 'backward':
        shifts = np.arange(-N + 1, 1, dtype=np.float64)
    elif strategy == 'center':
        if approx_order % 2 != 0:
            raise ValueError('Centered finite-difference requires an even order approximation.')
        N = num_points // 2
        shifts = np.arange(-N, N + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown strategy {strategy}. Must be one of 'forward', 'backward', 'center'.")
    A = shifts ** np.arange(len(shifts)).reshape(-1, 1)
    b = np.zeros_like(shifts)
    b[n] = factorial(n)
    coeffs = linalg_solve(A, b)
    coeffs_and_shifts = np.stack([coeffs, shifts])
    coeffs_and_shifts[np.abs(coeffs_and_shifts) < 1e-10] = 0
    coeffs_and_shifts = coeffs_and_shifts[:, ~np.all(coeffs_and_shifts == 0, axis=0)]
    coeffs_and_shifts = coeffs_and_shifts[:, np.argsort(np.abs(coeffs_and_shifts)[1])]
    return coeffs_and_shifts