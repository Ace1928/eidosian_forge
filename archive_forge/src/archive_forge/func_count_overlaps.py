import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def count_overlaps(self, bboxes):
    """
        Count the number of bounding boxes that overlap this one.

        Parameters
        ----------
        bboxes : sequence of `.BboxBase`
        """
    return count_bboxes_overlapping_bbox(self, np.atleast_3d([np.array(x) for x in bboxes]))