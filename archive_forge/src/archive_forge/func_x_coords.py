from .. import utils
from .scatter import _ScatterParams
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numpy as np
import pandas as pd
@property
def x_coords(self):
    self.x_labels
    return self._x_coords[self.plot_idx]