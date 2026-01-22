from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
@property
def discrete(self):
    """Check if the color array is discrete.

        If not provided:
        * If c is constant or an array, return None
        * If cmap is a dict, return True
        * If c has 20 or less unique values, return True
        * Otherwise, return False
        """
    if self._discrete is not None:
        return self._discrete
    elif self.constant_c() or self.array_c():
        return None
    elif isinstance(self._cmap, dict) or not np.all([isinstance(x, numbers.Number) for x in self._c_masked]):
        return True
    elif self.n_c_unique > 20:
        return False
    else:
        return np.allclose(self.c_unique % 1, 0, atol=0.0001)