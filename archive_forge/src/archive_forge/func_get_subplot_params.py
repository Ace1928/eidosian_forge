import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def get_subplot_params(self, figure=None):
    """Return a dictionary of subplot layout parameters."""
    hspace = self._hspace if self._hspace is not None else figure.subplotpars.hspace if figure is not None else mpl.rcParams['figure.subplot.hspace']
    wspace = self._wspace if self._wspace is not None else figure.subplotpars.wspace if figure is not None else mpl.rcParams['figure.subplot.wspace']
    figbox = self._subplot_spec.get_position(figure)
    left, bottom, right, top = figbox.extents
    return mpl.figure.SubplotParams(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)