import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
def transform_path_non_affine(self, path):
    ipath = path.interpolated(self._resolution)
    return Path(self.transform(ipath.vertices), ipath.codes)