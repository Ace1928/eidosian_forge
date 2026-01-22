from .._lazyload import matplotlib as mpl
from . import tools
import numpy as np
def tab40():
    """Create a discrete colormap with 40 unique colors.

    This colormap combines `matplotlib`'s `tab20b` and `tab20c` colormaps

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    return mpl.colors.ListedColormap(colors)