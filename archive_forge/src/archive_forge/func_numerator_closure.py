import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def numerator_closure(self):
    """The bridge closure, where consecutive pairs of strands at both the top and
        at the bottom are respectively joined by caps and cups. The numbers of
        strands at both the top and the bottom must be even. Returns a Link.

        A synonym for this is ``Tangle.bridge_closure()``.

        sage: BraidTangle([2,-1,2],4).numerator_closure().alexander_polynomial()
        t^2 - t + 1
        sage: BraidTangle([1,1,1]).rotate(1).numerator_closure().alexander_polynomial()
        t^2 - t + 1
        """
    m, n = self.boundary
    if m % 2 or n % 2:
        raise ValueError('To do bridge closure, both the top and bottom must have an even number of strands')
    T = self.copy()
    for i in range(0, m, 2):
        join_strands(T.adjacent[i], T.adjacent[i + 1])
    for i in range(0, n, 2):
        join_strands(T.adjacent[m + i], T.adjacent[m + i + 1])
    return Link(T.crossings, check_planarity=False)