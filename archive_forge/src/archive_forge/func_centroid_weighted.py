import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
@property
def centroid_weighted(self):
    ctr = self.centroid_weighted_local
    return tuple((idx + slc.start * spc for idx, slc, spc in zip(ctr, self.slice, self._spacing)))