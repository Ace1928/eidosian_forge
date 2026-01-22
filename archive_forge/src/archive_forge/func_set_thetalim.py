import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
def set_thetalim(self, *args, **kwargs):
    """
        Set the minimum and maximum theta values.

        Can take the following signatures:

        - ``set_thetalim(minval, maxval)``: Set the limits in radians.
        - ``set_thetalim(thetamin=minval, thetamax=maxval)``: Set the limits
          in degrees.

        where minval and maxval are the minimum and maximum limits. Values are
        wrapped in to the range :math:`[0, 2\\pi]` (in radians), so for example
        it is possible to do ``set_thetalim(-np.pi / 2, np.pi / 2)`` to have
        an axis symmetric around 0. A ValueError is raised if the absolute
        angle difference is larger than a full circle.
        """
    orig_lim = self.get_xlim()
    if 'thetamin' in kwargs:
        kwargs['xmin'] = np.deg2rad(kwargs.pop('thetamin'))
    if 'thetamax' in kwargs:
        kwargs['xmax'] = np.deg2rad(kwargs.pop('thetamax'))
    new_min, new_max = self.set_xlim(*args, **kwargs)
    if abs(new_max - new_min) > 2 * np.pi:
        self.set_xlim(orig_lim)
        raise ValueError('The angle range must be less than a full circle')
    return tuple(np.rad2deg((new_min, new_max)))