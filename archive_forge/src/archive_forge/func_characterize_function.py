import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def characterize_function(breakpoints, values):
    """
    Characterizes a piecewise linear function described by a
    list of breakpoints and function values.

    Args:
        breakpoints (list): The list of breakpoints of the
            piecewise linear function. It is assumed that
            the list of breakpoints is in non-decreasing
            order.
        values (list): The values of the piecewise linear
            function corresponding to the breakpoints.

    Returns:
        (int, list): a function characterization code and
            the list of slopes.

    .. note::
        The function characterization codes are

          * 1: affine
          * 2: convex
          * 3: concave
          * 4: step
          * 5: other

        If the function has step points, some of the slopes
        may be :const:`None`.
    """
    if not is_nondecreasing(breakpoints):
        raise ValueError('The list of breakpoints must be nondecreasing')
    step = False
    slopes = []
    for i in range(1, len(breakpoints)):
        if breakpoints[i] != breakpoints[i - 1]:
            slope = float(values[i] - values[i - 1]) / (breakpoints[i] - breakpoints[i - 1])
        else:
            slope = None
            step = True
        slopes.append(slope)
    if step:
        return (characterize_function.step, slopes)
    elif is_constant(slopes):
        return (characterize_function.affine, slopes)
    elif is_nondecreasing(slopes):
        return (characterize_function.convex, slopes)
    elif is_nonincreasing(slopes):
        return (characterize_function.concave, slopes)
    else:
        return (characterize_function.other, slopes)