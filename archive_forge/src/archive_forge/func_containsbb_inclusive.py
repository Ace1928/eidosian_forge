from param.parameterized import get_occupied_slots
from .util import datetime_types
def containsbb_inclusive(self, x):
    """
        Returns true if the given BoundingBox x is contained within the
        bounding box, including cases of exact match.
        """
    left, bottom, right, top = self.aarect().lbrt()
    leftx, bottomx, rightx, topx = x.aarect().lbrt()
    return left <= leftx and bottom <= bottomx and (right >= rightx) and (top >= topx)