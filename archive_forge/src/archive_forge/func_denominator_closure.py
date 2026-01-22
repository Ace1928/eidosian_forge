import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def denominator_closure(self):
    """The braid closure, where corresponding strands between the top and bottom
        are joined. The number of strands at the top must equal the number of strands at
        the bottom. Returns a Link.

        A synonym for this is ``Tangle.braid_closure()``.

        sage: BraidTangle([1,1,1]).braid_closure().alexander_polynomial()
        t^2 - t + 1
        sage: BraidTangle([1,-2,1,-2]).braid_closure().alexander_polynomial()
        t^2 - 3*t + 1
        >>> BraidTangle([1,-2,1,-2]).braid_closure().exterior().identify() # doctest: +SNAPPY
        [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]
        """
    m, n = self.boundary
    if m != n:
        raise ValueError('To do braid closure, both the top and bottom numbers of strands must be equal')
    T = self.copy()
    for i in range(n):
        join_strands(T.adjacent[i], T.adjacent[m + i])
    return Link(T.crossings, check_planarity=False)