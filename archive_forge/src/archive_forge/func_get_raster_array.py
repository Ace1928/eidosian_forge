import inspect
import re
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib import (
from matplotlib.colors import Normalize, cnames
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Path, PathPatch
from matplotlib.rcsetup import validate_fontsize, validate_fonttype, validate_hatch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from packaging.version import Version
from ...core.util import arraylike_types, cftime_types, is_number
from ...element import RGB, Polygons, Raster
from ..util import COLOR_ALIASES, RGB_HEX_REGEX
def get_raster_array(image):
    """
    Return the array data from any Raster or Image type
    """
    if isinstance(image, RGB):
        rgb = image.rgb
        data = np.dstack([np.flipud(rgb.dimension_values(d, flat=False)) for d in rgb.vdims])
    else:
        data = image.dimension_values(2, flat=False)
        if type(image) is Raster:
            data = data.T
        else:
            data = np.flipud(data)
    return data