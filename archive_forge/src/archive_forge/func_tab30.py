from .._lazyload import matplotlib as mpl
from . import tools
import numpy as np
def tab30():
    """Create a discrete colormap with 30 unique colors.

    This colormap combines `matplotlib`'s `tab20b` and `tab20c` colormaps,
    removing the lightest color of each hue.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    select_idx = np.repeat(np.arange(10), 3) * 4 + np.tile(np.arange(3), 10)
    return mpl.colors.ListedColormap(colors[select_idx])