import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def shape_on_sheet(self):
    """Return the shape of the array of the Slice on its sheet."""
    return (self[1] - self[0], self[3] - self[2])