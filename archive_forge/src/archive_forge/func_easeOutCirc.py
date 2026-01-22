from __future__ import division
import math
def easeOutCirc(n):
    """A circular tween function that begins fast and then decelerates.

    Args:
      n (int, float): The time progress, starting at 0.0 and ending at 1.0.

    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    n -= 1
    return math.sqrt(1 - n * n)