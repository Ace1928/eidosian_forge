import math
import cupy
def float_factorial(n):
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else cupy.inf