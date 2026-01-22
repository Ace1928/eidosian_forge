from collections import namedtuple
import math
import warnings
@property
def is_conformal(self) -> bool:
    """True if the transform is conformal.

        i.e., if angles between points are preserved after applying the
        transform, within rounding limits.  This implies that the
        transform has no effective shear.
        """
    a, b, c, d, e, f, g, h, i = self
    return abs(a * b + d * e) < self.precision