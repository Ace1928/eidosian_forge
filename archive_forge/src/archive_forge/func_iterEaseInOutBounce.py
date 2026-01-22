from __future__ import division
import math
def iterEaseInOutBounce(startX, startY, endX, endY, intervalSize):
    """Returns an iterator of a easeInOutBounce tween between the start and end points, incrementing the
    interpolation factor by intervalSize each time. Guaranteed to return the point for 0.0 first
    and 1.0 last no matter the intervalSize."""
    return iter(_iterTween(startX, startY, endX, endY, intervalSize, easeInOutBounce))