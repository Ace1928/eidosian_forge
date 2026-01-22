from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def modulatedBarData(length=768, width=32):
    """ 
    Returns an NumPy array that represents a modulated color bar ranging from 0 to 1.
    This is used to judge the perceived variation of the color gradient.
    
    Parameters
    ----------
    length: int
        Length of the data set. Values will vary from 0 to 1 over this axis.
    width: int
        Width of the data set. The modulation will vary from 0% to 4% over this axis.    
    """
    gradient = np.linspace(0.0, 1.0, length)
    modulation = -0.04 * np.sin(np.pi / 4 * np.arange(length))
    data = np.zeros((length, width))
    for idx in range(width):
        data[:, idx] = gradient + idx / (width - 1) * modulation
    clip_array(data, 0.0, 1.0, out=data)
    return data