import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
@_preprocess_data(replace_names=['x', 't'])
def funcy(ax, x, y, z, t=None):
    """
        Parameters
        ----------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        """