from __future__ import division
import math
def easeInOutQuart(n):
    """Accelerates, reaches the midpoint, and then decelerates. (Quartic function.)

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n *= 2
    if n < 1:
        return 0.5 * n ** 4
    else:
        n -= 2
        return -0.5 * (n ** 4 - 2)