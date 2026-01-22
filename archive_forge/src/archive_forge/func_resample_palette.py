import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def resample_palette(palette, ncolors, categorical, cmap_categorical):
    """
    Resample the number of colors in a palette to the selected number.
    """
    if len(palette) != ncolors:
        if categorical and cmap_categorical:
            palette = [palette[i % len(palette)] for i in range(ncolors)]
        else:
            lpad, rpad = (-0.5, 0.49999999999)
            indexes = np.linspace(lpad, len(palette) - 1 + rpad, ncolors)
            palette = [palette[int(np.round(v))] for v in indexes]
    return palette