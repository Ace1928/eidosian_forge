import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def get_height_ratios(self):
    """
        Return the height ratios.

        This is *None* if no height ratios have been set explicitly.
        """
    return self._row_height_ratios