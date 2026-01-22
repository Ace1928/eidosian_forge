from collections import namedtuple
import math
import warnings
@property
def is_orthonormal(self) -> bool:
    """True if the transform is orthonormal.

        Which means that the transform represents a rigid motion, which
        has no effective scaling or shear. Mathematically, this means
        that the axis vectors of the transform matrix are perpendicular
        and unit-length.  Applying an orthonormal transform to a shape
        always results in a congruent shape.
        """
    a, b, c, d, e, f, g, h, i = self
    return self.is_conformal and abs(1.0 - (a * a + d * d)) < self.precision and (abs(1.0 - (b * b + e * e)) < self.precision)