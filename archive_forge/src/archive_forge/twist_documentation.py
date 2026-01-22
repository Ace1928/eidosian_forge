from .links import CrossingStrand
from ..graphs import CyclicList

    Changes crossings so that no bigon permits a Type II move cancelling
    the pair of crossings at the end of the bigon.  The system is that the
    end crossing with the lowest label "wins" in determining the type of the twist
    region.  The code assumes that no link component is effectively a meridian
    loop for another component; said differently, no two bigons share a
    common edge.

    Note, this implementation fails if given something like a (2, n) torus link.
    