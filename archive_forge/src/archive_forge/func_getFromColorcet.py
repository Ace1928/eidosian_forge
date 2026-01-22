from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def getFromColorcet(name):
    """ Generates a ColorMap object from a colorcet definition. Same as ``colormap.get(name, source='colorcet')``. """
    try:
        import colorcet
    except ModuleNotFoundError:
        return None
    color_strings = colorcet.palette[name]
    color_list = []
    for hex_str in color_strings:
        if hex_str[0] != '#':
            continue
        if len(hex_str) != 7:
            raise ValueError(f"Invalid color string '{hex_str}' in colorcet import.")
        color_tuple = tuple(bytes.fromhex(hex_str[1:]))
        color_list.append(color_tuple)
    if len(color_list) == 0:
        return None
    cmap = ColorMap(name=name, pos=np.linspace(0.0, 1.0, len(color_list)), color=color_list)
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap