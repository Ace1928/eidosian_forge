from __future__ import division
import math
def easeOutPoly(n, degree=2):
    """Starts fast and decelerates to stop. (Polynomial function with custom degree.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.
      degree (int, float): The degree of the polynomial function.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    if not isinstance(degree, (int, float)) or degree < 0:
        raise ValueError('degree argument must be a positive number.')
    return 1 - abs((n - 1) ** degree)