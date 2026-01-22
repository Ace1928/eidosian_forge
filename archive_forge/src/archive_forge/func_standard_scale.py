import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import (
@staticmethod
def standard_scale(data2d, axis=1):
    """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
    if axis == 1:
        standardized = data2d
    else:
        standardized = data2d.T
    subtract = standardized.min()
    standardized = (standardized - subtract) / (standardized.max() - standardized.min())
    if axis == 1:
        return standardized
    else:
        return standardized.T