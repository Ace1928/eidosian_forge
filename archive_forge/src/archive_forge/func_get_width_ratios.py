import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def get_width_ratios(self):
    """
        Return the width ratios.

        This is *None* if no width ratios have been set explicitly.
        """
    return self._col_width_ratios