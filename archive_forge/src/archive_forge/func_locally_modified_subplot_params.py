import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
def locally_modified_subplot_params(self):
    """
        Return a list of the names of the subplot parameters explicitly set
        in the GridSpec.

        This is a subset of the attributes of `.SubplotParams`.
        """
    return [k for k in self._AllowedKeys if getattr(self, k)]