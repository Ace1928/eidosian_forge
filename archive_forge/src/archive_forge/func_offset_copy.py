import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    """
    Return a new transform with an added offset.

    Parameters
    ----------
    trans : `Transform` subclass
        Any transform, to which offset will be applied.
    fig : `~matplotlib.figure.Figure`, default: None
        Current figure. It can be None if *units* are 'dots'.
    x, y : float, default: 0.0
        The offset to apply.
    units : {'inches', 'points', 'dots'}, default: 'inches'
        Units of the offset.

    Returns
    -------
    `Transform` subclass
        Transform with applied offset.
    """
    _api.check_in_list(['dots', 'points', 'inches'], units=units)
    if units == 'dots':
        return trans + Affine2D().translate(x, y)
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')
    if units == 'points':
        x /= 72.0
        y /= 72.0
    return trans + ScaledTranslation(x, y, fig.dpi_scale_trans)