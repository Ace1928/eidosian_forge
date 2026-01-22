import math
from typing import NamedTuple
from dataclasses import dataclass
def reverseTransform(self, other):
    """Return a new transformation, which is the other transformation
        transformed by self. self.reverseTransform(other) is equivalent to
        other.transform(self).

        :Example:
                >>> t = Transform(2, 0, 0, 3, 1, 6)
                >>> t.reverseTransform((4, 3, 2, 1, 5, 6))
                <Transform [8 6 6 3 21 15]>
                >>> Transform(4, 3, 2, 1, 5, 6).transform((2, 0, 0, 3, 1, 6))
                <Transform [8 6 6 3 21 15]>
                >>>
        """
    xx1, xy1, yx1, yy1, dx1, dy1 = self
    xx2, xy2, yx2, yy2, dx2, dy2 = other
    return self.__class__(xx1 * xx2 + xy1 * yx2, xx1 * xy2 + xy1 * yy2, yx1 * xx2 + yy1 * yx2, yx1 * xy2 + yy1 * yy2, xx2 * dx1 + yx2 * dy1 + dx2, xy2 * dx1 + yy2 * dy1 + dy2)