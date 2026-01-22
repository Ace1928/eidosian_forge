from numbers import Number
import functools
from types import MethodType
import numpy as np
from matplotlib import _api, cbook
from matplotlib.gridspec import SubplotSpec
from .axes_divider import Size, SubplotDivider, Divider
from .mpl_axes import Axes, SimpleAxisArtist
@_api.deprecated('3.8', alternative='ax.tick_params and colorbar.set_label')
def toggle_label(self, b):
    axis = self.axis[self.orientation]
    axis.toggle(ticklabels=b, label=b)