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
def process_string_cmap(self, cmap):
    """Subset a discrete colormap based on the number of colors if necessary."""
    cmap = mpl.cm.get_cmap(cmap)
    if self.discrete and cmap.N <= 20 and (self.n_c_unique <= cmap.N):
        return mpl.colors.ListedColormap(cmap.colors[:self.n_c_unique])
    else:
        return cmap