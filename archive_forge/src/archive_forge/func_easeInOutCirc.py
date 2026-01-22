from __future__ import division
import math
def easeInOutCirc(n):
    """A circular tween function that accelerates, reaches the midpoint, and then decelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n *= 2
    if n < 1:
        return -0.5 * (math.sqrt(1 - n ** 2) - 1)
    else:
        n -= 2
        return 0.5 * (math.sqrt(1 - n ** 2) + 1)