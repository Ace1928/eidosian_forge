from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def listMaps(source=None):
    """
    .. warning:: Experimental, subject to change.

    List available color maps.

    Parameters
    ----------
    source: str, optional
        Color map source. If omitted, locally stored maps are listed. Otherwise:

          - 'matplotlib' lists maps that can be imported from Matplotlib
          - 'colorcet' lists maps that can be imported from ColorCET

    Returns
    -------
    list of str
        Known color map names.
    """
    if source is None:
        pathname = path.join(path.dirname(__file__), 'colors', 'maps')
        files = listdir(pathname)
        list_of_maps = []
        for filename in files:
            if filename[-4:] == '.csv' or filename[-4:] == '.hex':
                list_of_maps.append(filename[:-4])
        return list_of_maps
    elif source.lower() == 'matplotlib':
        try:
            import matplotlib.pyplot as mpl_plt
            list_of_maps = mpl_plt.colormaps()
            return list_of_maps
        except ModuleNotFoundError:
            return []
    elif source.lower() == 'colorcet':
        try:
            import colorcet
            list_of_maps = list(colorcet.palette.keys())
            list_of_maps.sort()
            return list_of_maps
        except ModuleNotFoundError:
            return []
    return []