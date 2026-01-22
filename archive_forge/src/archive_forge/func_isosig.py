import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def isosig(self, root=None, over_or_under=False):
    """
        Return a bunch of data which encodes the planar isotopy class of the
        tangle.  Of course, this is just up to isotopy of the plane
        (no Reidemeister moves).  A root can be specified with a CrossingStrand
        and ``over_or_under`` toggles whether only the underlying
        shadow (4-valent planar map) is considered or the tangle with the
        over/under data at each crossing.

        >>> BraidTangle([1]).isosig() == BraidTangle([1]).circular_rotate(1).isosig()
        True
        >>> BraidTangle([1]).isosig() == BraidTangle([-1]).isosig()
        True
        """
    copy = self.copy()
    copy._fuse_strands()
    return planar_isotopy.min_isosig(copy, root, over_or_under)