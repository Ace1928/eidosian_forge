import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
@property
def rowspan(self):
    """The rows spanned by this subplot, as a `range` object."""
    ncols = self.get_gridspec().ncols
    return range(self.num1 // ncols, self.num2 // ncols + 1)