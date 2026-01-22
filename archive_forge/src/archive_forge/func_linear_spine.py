from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
@classmethod
def linear_spine(cls, axes, spine_type, **kwargs):
    """Create and return a linear `Spine`."""
    if spine_type == 'left':
        path = mpath.Path([(0.0, 0.999), (0.0, 0.999)])
    elif spine_type == 'right':
        path = mpath.Path([(1.0, 0.999), (1.0, 0.999)])
    elif spine_type == 'bottom':
        path = mpath.Path([(0.999, 0.0), (0.999, 0.0)])
    elif spine_type == 'top':
        path = mpath.Path([(0.999, 1.0), (0.999, 1.0)])
    else:
        raise ValueError('unable to make path for spine "%s"' % spine_type)
    result = cls(axes, spine_type, path, **kwargs)
    result.set_visible(mpl.rcParams[f'axes.spines.{spine_type}'])
    return result