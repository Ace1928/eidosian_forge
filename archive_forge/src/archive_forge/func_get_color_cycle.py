import os
from colorcet import kbc, register_cmap
from matplotlib import rc_params_from_file
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from packaging.version import Version
from param import concrete_descendents
from ...core import Collator, GridMatrix, Layout, config
from ...core.options import Cycle, Options, Palette
from ...core.overlay import NdOverlay, Overlay
from ...element import *
from ..plot import PlotSelector
from ..util import fire_colors
from .annotation import *
from .chart import *
from .chart3d import *
from .element import ElementPlot
from .geometry import *
from .graphs import *
from .heatmap import *
from .hex_tiles import *
from .path import *
from .plot import *
from .raster import *
from .renderer import MPLRenderer
from .sankey import *
from .stats import *
from .tabular import *
def get_color_cycle():
    if mpl_ge_150:
        cyl = mpl.rcParams['axes.prop_cycle']
        try:
            return [x['color'] for x in cyl]
        except KeyError:
            pass
    return mpl.rcParams['axes.color_cycle']